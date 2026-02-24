# Copyright (c) [2025] [FastVideo Team]
# Copyright (c) [2025] [ByteDance Ltd. and/or its affiliates.]
# Copyright (C) 2025 Tencent.
# SPDX-License-Identifier: [Apache License 2.0] 
#
# This file has been modified by [ByteDance Ltd. and/or its affiliates.] in 2025.
#
# Original file was released under [Apache License 2.0], with the full license text
# available at [https://github.com/hao-ai-lab/FastVideo/blob/main/LICENSE].
#
# This modified file is released under the same license.
import sys
import math
import pdb
from transformers import AltCLIPModel, AltCLIPProcessor
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import Compose, Resize, CenterCrop, Normalize
import PIL.Image as Image
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC
import logging
from torchvision.transforms import Compose, Resize, CenterCrop, Normalize
from loguru import logger
import argparse
import os
import pickle
from safetensors.torch import load_file
import torchvision.transforms  as transforms

from fastvideo.utils.parallel_states import (
    initialize_sequence_parallel_state,
    destroy_sequence_parallel_group,
    get_sequence_parallel_state,
    nccl_info,
)
import random
from fastvideo.utils.communications import sp_parallel_dataloader_wrapper
from torch.utils.data import DataLoader
import torch

from torch import nn
from torch.utils.data.distributed import DistributedSampler

from accelerate.utils import set_seed
from tqdm.auto import tqdm
from fastvideo.utils.fsdp_util import get_dit_fsdp_kwargs, apply_fsdp_checkpointing
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version
from fastvideo.dataset.latent_qwenimage_rl_datasets import LatentDataset, latent_collate_function
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from fastvideo.utils.checkpoint import (
    save_checkpoint,
    save_checkpoint_bf16,
    save_lora_checkpoint,
)
from fastvideo.utils.logging_ import main_print
from diffusers.image_processor import VaeImageProcessor
from transformers import AutoProcessor, AutoModel

check_min_version("0.31.0")
import time
from collections import deque
import torch.distributed as dist
from torch.nn import functional as F
# from diffusers import FluxTransformer2DModel, AutoencoderKL
from fastvideo.utils.fsdp_util_qwenimage import fsdp_wrapper, FSDPConfig

from hpsv2.src.open_clip import create_model_and_transforms, get_tokenizer
def round_to_base(x, base):
    return max(base, int(round(x / base)) * base)

def generate_candidates(base, max_side=2048):
    base_widths = [3840, 2560, 1920, 1600, 1360, 1280, 1024, 896, 768, 640, 512, 384]
    ARs = [16/9, 4/3, 1/1, 3/4, 9/16,3/2,2/3] 
    cands = set()
    for w0 in base_widths:
        if w0 > max_side:
            continue
        for ar in ARs:
            w = round_to_base(w0, base)
            h = round_to_base(w0 / ar, base)
            if w <= max_side and h <= max_side:
                cands.add((w, h))
            h_alt = round_to_base(w0, base)
            w_alt = round_to_base(w0 * ar, base)
            if w_alt <= max_side and h_alt <= max_side:
                cands.add((w_alt, h_alt))
    smalls = [(base, base), (base*2, base), (base*2, base*2), (1280,720), (1024,768), (720,1280)]
    for s in smalls:
        if s[0] <= max_side and s[1] <= max_side:
            cands.add(s)
    return sorted(list(cands), key=lambda x: x[0]*x[1])

def choose_best_resolution(w, h, SPATIAL_DOWNSAMPLE=8, divisible_by=8, max_side=2048, allow_upscale=False):
    base = SPATIAL_DOWNSAMPLE * divisible_by
    ar_in = w / h
    area_in = w * h

    candidates = generate_candidates(base, max_side=max_side)
    if not candidates:

        tw = min(round_to_base(w, base), max_side)
        th = min(round_to_base(h, base), max_side)
        return tw, th

    best = None
    best_score = float('inf')
    for cw, ch in candidates:
        ar_c = cw / ch
        area_c = cw * ch
        aspect_diff = abs(ar_in - ar_c) / max(ar_in, ar_c)
        area_ratio = area_c / area_in

        if not allow_upscale and area_ratio > 1.0:

            area_penalty = abs(math.log(area_ratio)) * 1.0 + 1.0
        else:
            area_penalty = abs(math.log(max(area_ratio, 1e-8))) * 0.5

        score = aspect_diff * 1.0 + area_penalty * 0.6


        if area_ratio > 4.0 or area_ratio < 0.1:
            continue

        if score < best_score:
            best_score = score
            best = (cw, ch)

    if best is None:

        tw = min(round_to_base(w, base), max_side)
        th = min(round_to_base(h, base), max_side)
        return tw, th

    return best

def random_resolution(step=1600):
    def adjust_resolution(res_list):
        adjusted = []
        for w, h in res_list:
            w_adj = w - (w % 16)
            h_adj = h - (h % 16)
            adjusted.append((w_adj, h_adj))
        return adjusted

    resolutions = [
        (640, 480),
        (800, 600),
        (1024, 768),
        (1280, 720),
        (1280, 1280),
        (1280, 800),
        (1366, 768),
        (1440, 900),
        (1024, 1024),
        (1600, 900),
        (1600, 900),
    ]

    adjusted_resolutions = adjust_resolution(resolutions)

    chosen = adjusted_resolutions[step%len(adjusted_resolutions)]
    return chosen

def get_random_cg_oily_adjective(index = 0):
    cg_oily_adjectives = [
        "Concept art",
        "Painting",
        "Anime",

    ]
    return cg_oily_adjectives[index%len(cg_oily_adjectives)]

# Some texture word
def get_random_realism_adjective(index = 0):
    realism_adjectives = [
      "Natural-lighting","Detail","Detailed","Real"
    ]
    return   realism_adjectives[index%len(realism_adjectives)]

class CLIP(torch.nn.Module):
    def __init__(self,is_pickscore=True,device="cuda", dtype=torch.float32):
        super().__init__()
        processor_path = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
        model_path = "yuvalkirstain/PickScore_v1"
        self.device = device
        self.dtype = dtype
        if is_pickscore:
            self.processor = AutoProcessor.from_pretrained(processor_path)
            self.model = AutoModel.from_pretrained(model_path).eval().to(device)
        else:
            self.processor = AutoProcessor.from_pretrained(processor_path)
            self.model = AutoModel.from_pretrained(processor_path).eval().to(device)
        self.model = self.model.to(dtype=dtype)
        image_mean = (0.48145466, 0.4578275, 0.40821073)
        image_std = (0.26862954, 0.26130258, 0.27577711)
        crop_size = 224
        resize_size = 224  
        def _transform():
            transform = Compose([
                Resize(resize_size, interpolation=BICUBIC),
                CenterCrop(crop_size),
                Normalize(std=image_std,mean=image_mean),
            ])
            return transform
        self.v_pre = _transform()

    #### implment the SRP in CFG-like function；we find the （1-k)*neg + k *pos is less stable, we change it to (1+k)*pos-neg
    def SRP_cfg(self, prompt, neg_prompt, image_inputs,k):
        # Extract image features and text features for positive and negative prompts
        image_inputs = self.v_pre(image_inputs)
        text_inputs = self.processor(
            text=prompt,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        )
        text_inputs = {k: v.to(device=self.device) for k, v in text_inputs.items()}
       
        neg_text_input = self.processor(
            text=neg_prompt,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        )
        neg_text_input = {k: v.to(device=self.device) for k, v in neg_text_input.items()}
        image_embs = self.model.get_image_features(pixel_values=image_inputs)
        image_embs = image_embs / image_embs.norm(p=2, dim=-1, keepdim=True)
       
        # Extract image features and text features for positive and negative prompts
        text_embs = self.model.get_text_features(**text_inputs)
        text_embs = text_embs / text_embs.norm(p=2, dim=-1, keepdim=True)
        text_embs_neg = self.model.get_text_features(**neg_text_input)
        text_embs_neg = text_embs_neg / text_embs_neg.norm(p=2, dim=-1, keepdim=True)

        logit_scale = self.model.logit_scale.exp()

        # Compute the reward based on the similarity
        scores = logit_scale * ((k+1)*text_embs-text_embs_neg)@ image_embs.T
        scores = scores.diag()
        #Make score-scope compareable with hps
        scores = scores/20
        return scores

class AltCLIPRM(nn.Module):
    def __init__(self, resize_res=224):
        super().__init__()
        self.model = AltCLIPModel.from_pretrained('path2alt-clip', dtype=torch.bfloat16)

        self.processor = AltCLIPProcessor.from_pretrained('BAAI/AltCLIP', use_fast=False)
        self.tokenizer = self.processor.tokenizer
        image_mean = (0.48145466, 0.4578275, 0.40821073)
        image_std = (0.26862954, 0.26130258, 0.27577711)
        crop_size = 512
        resize_size = 512  
        def _transform():
            transform = Compose([
                Resize(resize_size, interpolation=BICUBIC),
                CenterCrop(crop_size),
                Normalize(std=image_std,mean=image_mean),
            ])
            return transform
        self.image_processor = _transform()

    def get_text_features(self, *args, **kwargs):
        return self.model.get_text_features(*args, **kwargs)


    def get_image_features(self, *args, **kwargs):
        return self.model.get_image_features(*args, **kwargs)


    def forward(self, text_inputs=None, image_inputs=None):
        outputs = ()
        if text_inputs is not None:
            outputs += self.model.get_text_features(**text_inputs),
        if image_inputs is not None:
            outputs += self.model.get_image_features(image_inputs, interpolate_pos_encoding=True),  # compatible with different input shapes
        return outputs


    @property
    def logit_scale(self):
        return self.model.logit_scale

    def load(self, path):
        self.model = AltCLIPModel.from_pretrained(path, dtype=torch.bfloat16, weights_only=False)

    def save(self, path):
        self.model.save_pretrained(path)

    def tokenize(self, prompt):
        text_inputs = self.tokenizer(
            prompt,
            max_length=self.tokenizer.model_max_length,
            # max_length=77,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return text_inputs

    def process_image(self, images): 
        pixel_values = self.image_processor(images.squeeze(0)).unsqueeze(0)
        return pixel_values

    def get_preference_scores(self, prompt, images):
        # print(self.processor.image_processor)
        image_inputs = self.process_image(images).to(self.model.device)
        text_inputs = self.tokenize(prompt).to(self.model.device)
        text_embs, image_embs = self.forward(text_inputs, image_inputs)
        
        image_embs = image_embs / torch.norm(image_embs, dim=-1, keepdim=True)
        text_embs = text_embs / torch.norm(text_embs, dim=-1, keepdim=True)

        rm_scores = self.logit_scale.exp() * (text_embs @ image_embs.T).flatten()
        
        # probs = torch.softmax(rm_scores, dim=-1)
        
        return rm_scores

    def SRP_cfg(self, prompt,prompt2, images,k):
        # print(self.processor.image_processor)
        image_inputs = self.process_image(images).to(self.model.device)
        text_inputs = self.tokenize(prompt).to(self.model.device)
        text_inputs2 = self.tokenize(prompt2).to(self.model.device)
        text_embs,image_embs = self.forward(text_inputs, image_inputs)
        text_embs2 =  self.model.get_text_features(**text_inputs2)
        image_embs = image_embs / torch.norm(image_embs, dim=-1, keepdim=True)
        text_embs = text_embs / torch.norm(text_embs, dim=-1, keepdim=True)
        text_embs2 = text_embs2 / torch.norm(text_embs2, dim=-1, keepdim=True)
        rm_scores = self.logit_scale.exp() * ( ((1+k)*text_embs-text_embs2) @ image_embs.T).flatten()/300
        
        # probs = torch.softmax(rm_scores, dim=-1)
        
        return rm_scores

class HPS(torch.nn.Module):
    def __init__(self, device="cuda", dtype=torch.float32):
        super().__init__()
        self.device = device
        self.dtype = dtype
        hpsv2_model,hpsv2_token,hpsv2_pre=self.build_reward_model()
        self.model = hpsv2_model.to(dtype=dtype)
        self.token = hpsv2_token 

        #### differentiable preprocessor
        image_mean = (0.48145466, 0.4578275, 0.40821073)
        image_std = (0.26862954, 0.26130258, 0.27577711)
        crop_size = 224
        resize_size = 224 
        def _transform():
            transform = Compose([
                Resize(resize_size, interpolation=BICUBIC),
                CenterCrop(crop_size),

                Normalize(std=image_std,mean=image_mean),
            ])
            return transform
        self.vis_pre = _transform()
        self.device =device

    def build_reward_model(self):
        model, preprocess_train, reprocess_val = create_model_and_transforms(
            'ViT-H-14',
            'laion2B-s32B-b79K',
            precision='amp',
            device=self.device,
            jit=False,
            force_quick_gelu=False,
            force_custom_text=False,
            force_patch_dropout=False,
            force_image_size=None,
            pretrained_image=False,
            image_mean=None,
            image_std=None,
            light_augmentation=True,
            aug_cfg={},
            output_dict=True,
            with_score_predictor=False,
            with_region_predictor=False
        )

        # Convert device name to proper format
        if isinstance(self.device, int):
            ml_device = str(self.device)
        else:
            ml_device = self.device

        if not ml_device.startswith('cuda'):
            ml_device = f'cuda:{ml_device}' if ml_device.isdigit() else ml_device

        checkpoint = torch.load('./hps_ckpt/HPS_v2.1_compressed.pt', map_location=ml_device)
        model.load_state_dict(checkpoint['state_dict'])
        text_processor = get_tokenizer('ViT-H-14')
        reward_model = model.to(self.device)
        reward_model.eval()

        return reward_model, text_processor, preprocess_train

    #### implment the SRP in CFG-like function；we find the （1-k)*neg + k *pos is less stable. therefore, we change it to (1+k)*pos-neg
    def SRP_cfg(self, prompt,neg_prompt,images,k):
        image = self.vis_pre(images.squeeze(0)).unsqueeze(0).to(device=self.device, non_blocking=True)
        text = self.token(prompt).to(device=self.device, non_blocking=True)
        neg_text = self.token(neg_prompt).to(device=self.device, non_blocking=True)
        with torch.cuda.amp.autocast():
            # Extract image features and text features for positive and negative prompts
            image_features = self.model.encode_image(image, normalize=True)
            text_features = self.model.encode_text(text, normalize=True)
            text_features_neg = self.model.encode_text(neg_text, normalize=True)
            
            # Compute the reward based on the similarity
            logits_per_image = image_features @ ((1+k)*text_features.T-text_features_neg.T)
            hps_score = torch.diagonal(logits_per_image)
        return hps_score

    def SRP(self, prompt,images,k):
        image = self.vis_pre(images.squeeze(0)).unsqueeze(0).to(device=self.device, non_blocking=True)
        text = self.token(prompt).to(device=self.device, non_blocking=True)
        with torch.cuda.amp.autocast():
            image_features = self.model.encode_image(image, normalize=True)
            text_features = self.model.encode_text(text, normalize=True)
            logits_per_image = image_features @ (k*text_features.T)
            hps_score = torch.diagonal(logits_per_image)
        return hps_score

def sd3_time_shift(shift, t):
    return (shift * t) / (1 + (shift - 1) * t)

def empty_logger():
    logger = logging.getLogger("hymm_empty_logger")
    logger.addHandler(logging.NullHandler())
    logger.setLevel(logging.CRITICAL)
    return logger

def setup_logger(exp_dir):
    if int(os.environ["RANK"]) <= 0:
        logger.add(os.path.join(exp_dir, "train.log"), level="DEBUG", colorize=False, backtrace=True,
                   diagnose=True, encoding="utf-8", filter=logger_filter("train"))
        logger.add(os.path.join(exp_dir, "val.log"), level="DEBUG", colorize=False, backtrace=True,
                   diagnose=True, encoding="utf-8", filter=logger_filter("val"))
        train_logger = logger.bind(name="train")
        val_logger = logger.bind(name="val")
    else:
        val_logger = train_logger = empty_logger()

    train_logger.info(f"Experiment directory created at: {exp_dir}")

    return train_logger, val_logger
def flux_step(
    model_output: torch.Tensor,
    latents: torch.Tensor,
    eta: float,
    sigmas: torch.Tensor,
    index: int,
    prev_sample: torch.Tensor,
    grpo: bool,
    sde_solver: bool,
):
    sigma = sigmas[index]
    dsigma = sigmas[index + 1] - sigma
    prev_sample_mean = latents + dsigma * model_output
    return prev_sample_mean

def main_print(content):
    if int(os.environ["RANK"]) <= 0:
        print(content)

def empty_logger():
    logger = logging.getLogger("hymm_empty_logger")
    logger.addHandler(logging.NullHandler())
    logger.setLevel(logging.CRITICAL)
    return logger

def logger_filter(name):
    def filter_(record):
        return record["extra"].get("name") == name
    return filter_
def assert_eq(x, y, msg=None):
    assert x == y, f"{msg or 'Assertion failed'}: {x} != {y}"

def prepare_latent_image_ids(batch_size, height, width, device, dtype):
    latent_image_ids = torch.zeros(height, width, 3)
    latent_image_ids[..., 1] = latent_image_ids[..., 1] + torch.arange(height)[:, None]
    latent_image_ids[..., 2] = latent_image_ids[..., 2] + torch.arange(width)[None, :]

    latent_image_id_height, latent_image_id_width, latent_image_id_channels = latent_image_ids.shape

    latent_image_ids = latent_image_ids.reshape(
        latent_image_id_height * latent_image_id_width, latent_image_id_channels
    )

    return latent_image_ids.to(device=device, dtype=dtype)

def pack_latents(latents, batch_size, num_channels_latents, height, width):
    latents = latents.view(batch_size, num_channels_latents, height // 2, 2, width // 2, 2)
    latents = latents.permute(0, 2, 4, 1, 3, 5)
    latents = latents.reshape(batch_size, (height // 2) * (width // 2), num_channels_latents * 4)
    
    return latents

def unpack_latents(latents, height, width, vae_scale_factor):
    batch_size, num_patches, channels = latents.shape

    # VAE applies 8x compression on images but we must also account for packing which requires
    # latent height and width to be divisible by 2.
    height = 2 * (int(height) // (vae_scale_factor * 2))
    width = 2 * (int(width) // (vae_scale_factor * 2))

    latents = latents.view(batch_size, height // 2, width // 2, channels // 4, 2, 2)
    latents = latents.permute(0, 3, 1, 4, 2, 5)

    # latents = latents.reshape(batch_size, channels // (2 * 2), height, width)
    latents = latents.reshape(batch_size, channels // (2 * 2), 1, height, width)

    return latents


def run_sample_step(
        args,
        z,
        progress_bar,
        sigma_schedule,
        transformer,
        encoder_hidden_states, 
        prompt_attention_mask, 
        encoder_hidden_states_n, 
        prompt_attention_mask_n, 
       
        img_shapes,
        txt_seq_lens,
        cfg,
    ):

    for i in progress_bar: 
        B = encoder_hidden_states.shape[0]
        sigma = sigma_schedule[i]
        timestep_value = int(sigma * 1000)

        timesteps = torch.full([encoder_hidden_states.shape[0]], timestep_value, device=z.device, dtype=torch.long)
        transformer.eval()

        with torch.autocast("cuda", torch.bfloat16):

            pred = transformer(
                        hidden_states=z,
                        timestep=timesteps/1000,
                        guidance=None,
                        encoder_hidden_states_mask=prompt_attention_mask,
                        encoder_hidden_states=encoder_hidden_states,
                        img_shapes=img_shapes,
                        txt_seq_lens=txt_seq_lens,
                        attention_kwargs=None,
                        return_dict=False,
                    )[0]
            if not cfg ==1: 
                pred_un = transformer(
                    hidden_states=z,
                    timestep=timesteps/1000,
                    guidance=None,
                    encoder_hidden_states_mask=prompt_attention_mask_n,
                    encoder_hidden_states=encoder_hidden_states_n,
                    img_shapes=img_shapes,
                    txt_seq_lens=txt_seq_lens,
                    attention_kwargs=None,
                    return_dict=False,
                )[0]
                comb_pred = pred_un +  cfg * (pred - pred_un)
                cond_norm = torch.norm(pred, dim=-1, keepdim=True)
                noise_norm = torch.norm(comb_pred, dim=-1, keepdim=True)
                noise_pred = comb_pred * (cond_norm / noise_norm)
            else:
                noise_pred=pred
        z= flux_step(noise_pred, z.to(torch.float32), args.eta, sigmas=sigma_schedule, index=i, prev_sample=None, grpo=True, sde_solver=True)
        z.to(torch.bfloat16)
    return z

def SRPO_train(
    args,
    device, 
    transformer,
    vae,
    encoder_hidden_states, 
    prompt_attention_masks, 
    encoder_hidden_states_n, 
    prompt_attention_masks_n, 
    original_length,
    reward_model,
    caption,
    mid_timestep,
    step,
    fp,
    visualization_step,
):
    timestep_length = args.timestep_length
    t=1
    shift = args.shift
    w,h = args.w, args.h
    if not visualization_step:
        cfg = args.train_cfg
        sample_steps = args.sampling_steps
        gradient_accumulation_steps = args.gradient_accumulation_steps 
    else:
        cfg = 4
        gradient_accumulation_steps = 1 # useless for visualization
        sample_steps = args.vis_sampling_step
        h = w = args.vis_size
  
    sigma_schedule = torch.linspace(1, 0, sample_steps + 1)
    sigma_schedule = sd3_time_shift(shift, sigma_schedule)
    discount = torch.linspace(0, 1, timestep_length+1) 
    discount = sd3_time_shift(3, discount).to(device)
    image_processor = VaeImageProcessor(16)
    B = encoder_hidden_states.shape[0]
    SPATIAL_DOWNSAMPLE = 8
    IN_CHANNELS = 16
    

    batch_size = 1  
    batch_indices = torch.chunk(torch.arange(B), B // batch_size)
    gen = torch.Generator(device=f"cuda")

    latents_mean = (torch.tensor(vae.config.latents_mean).view(1, vae.config.z_dim, 1, 1, 1)).to(device)
    latents_std = 1.0 / torch.tensor(vae.config.latents_std).view(1, vae.config.z_dim, 1, 1, 1).to(device)

    Offline = True    
    groundtruth_ratio = args.groundtruth_ratio
    if step > args.offline_end:
        Offline=False
    if Offline:
        def load_image_tensor(path, size=(1024,1024)):
            image = Image.open(path).convert("RGB")
            w, h = image.size
            w, h = choose_best_resolution(w, h, SPATIAL_DOWNSAMPLE=8, divisible_by=8, max_side=1024, allow_upscale=False)
            image = image.resize((w, h), resample=Image.LANCZOS)
            preprocess = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]) 
            ])
            image = preprocess(image).unsqueeze(0)   # [1, 3, H, W]
            return image,w,h
        with open(args.offline_dict, 'rb') as f:
            image_d = pickle.load(f)
        image_path = image_d[caption[0]]

        image_start,w,h = load_image_tensor(image_path)
        image = image_start.to("cuda").unsqueeze(2)
 
        def encode_image_to_latents(pixel_values, vae) -> torch.Tensor:

            with torch.no_grad():
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    model_input = vae.encode(pixel_values).latent_dist.sample()
                    model_input = (model_input - latents_mean) * latents_std
            return model_input
        offline_latent = encode_image_to_latents(image,vae)
    latent_w, latent_h = w // SPATIAL_DOWNSAMPLE, h // SPATIAL_DOWNSAMPLE
    input_latents = torch.randn(
                    (1, 1, IN_CHANNELS, latent_h, latent_w),  #（c,t,h,w)
                    device=device,
                    dtype=torch.bfloat16,
                    generator=gen
                    )
    if not Offline:
        offline_latent = input_latents

    #### it is important to make all device generate with the same latent
    for index, batch_idx in enumerate(batch_indices):
        batch_encoder_hidden_states = encoder_hidden_states[batch_idx][:,:original_length[batch_idx]]
        batch_prompt_attention_mask = prompt_attention_masks[batch_idx][:,:original_length[batch_idx]]
        batch_encoder_hidden_states_n = encoder_hidden_states_n[batch_idx][:,:original_length[batch_idx]]
        batch_prompt_attention_mask_n = prompt_attention_masks_n[batch_idx][:,:original_length[batch_idx]]
        batch_caption = [caption[i] for i in batch_idx]
        packed_height = 2 * (int(h) // (8 * 2))
        packed_width = 2 * (int(w) // (8 * 2))
        online_latent_new = pack_latents(input_latents, len(batch_idx), 16, packed_height, packed_width)
        offline_latent_new = pack_latents(offline_latent, len(batch_idx), 16, packed_height, packed_width)
        txt_seq_lens =  batch_prompt_attention_mask.sum(dim=1).tolist() 
        img_shapes = [[(1, h // 8 // 2, w // 8// 2)]] 

        progress_bar = tqdm(range(0, sample_steps), desc="Sampling Progress")
        
      
        if  not Offline or  visualization_step:
            with torch.no_grad():
                latent_start = run_sample_step(
                    args,
                    online_latent_new,
                    progress_bar,
                    sigma_schedule,
                    transformer,
                    batch_encoder_hidden_states,
                    batch_prompt_attention_mask,
                    batch_encoder_hidden_states_n,
                    batch_prompt_attention_mask_n,
                    img_shapes,
                    txt_seq_lens,
                    4.0
                )
        else:
            latent_start = offline_latent_new
        if visualization_step: 

            with torch.inference_mode():
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    latents = unpack_latents(latent_start, h, w, 8)
                    latents = latents.to(vae.dtype)
                    latents_mean = (
                        torch.tensor(vae.config.latents_mean)
                        .view(1, vae.config.z_dim, 1, 1, 1)
                        .to(latents.device, latents.dtype)
                    )
                    latents_std = 1.0 / torch.tensor(vae.config.latents_std).view(1, vae.config.z_dim, 1, 1, 1).to(
                        latents.device, latents.dtype
                    )
                    latents = latents / latents_std + latents_mean
                    image = vae.decode(latents, return_dict=False)[0][:, :, 0]
                    decoded_image = image_processor.postprocess(image)
            try:
                if step<100:
                    decoded_image[0].save(fp+f"/{step}_{dist.get_rank()}.png")
                # pass
            except:
                pass
            return 0

        # add control word to the prompt to control reward prefenece
        pos_control = get_random_realism_adjective(step)
        neg_control  =  get_random_cg_oily_adjective(step)
        pos_caption = pos_control+'. '+batch_caption[0]
        batch_caption[0] = neg_control+'. '+batch_caption[0]


    
        k = int((1 - groundtruth_ratio) * timestep_length) + 1
        if Offline:
            noise = torch.randn_like(online_latent_new).to(device)
            dist.broadcast(noise, src=0)
        else:
            noise = online_latent_new   
    
        num_inference_steps = timestep_length
        sigma_schedule = torch.linspace(1, 0, args.vis_sampling_step + 1)
        sigma_schedule = sd3_time_shift(shift, sigma_schedule)  
  

      
        for i in reversed(range(gradient_accumulation_steps)):
            
            timestep_select = torch.tensor([random.randint(2, args.timestep_length-2)], device=device)
            k = min(min(timestep_length-timestep_select,k),timestep_select)


            #####Timestep for training########
            sigmas_l = torch.linspace(sigma_schedule[args.train_timestep[0]], sigma_schedule[args.train_timestep[1]], num_inference_steps).to(device)
           

            mid_timestep = timestep_select
            start = min(mid_timestep+k,num_inference_steps)
            t_base = num_inference_steps-mid_timestep
            t_start = num_inference_steps-start

            ############Direct-Ailgn step1 inject noise###############
            with torch.no_grad():
      
                sigmas = sigmas_l[t_start]
                noisy = sigmas * noise + (1.0 - sigmas) * latent_start

                sigmast = sigmas_l[t_base]    
                gt_vector = sigmast * noise
            latents = noisy.detach()
            transformer.train()
            latents = latents.requires_grad_(True)

            ############Direct-Ailgn step2 denoise one step###############

            with torch.autocast("cuda", torch.bfloat16):
                sigma = sigmas_l[t_start]
                timestep_value = int(sigma * 1000)
                timesteps = torch.full([batch_encoder_hidden_states.shape[0]], timestep_value, device=latent_start.device, dtype=torch.long)
                pred = transformer(
                        hidden_states=latents,
                        timestep=timesteps/1000,
                        guidance=None,
                        encoder_hidden_states_mask=batch_prompt_attention_mask,
                        encoder_hidden_states=batch_encoder_hidden_states,
                        img_shapes=img_shapes,
                        txt_seq_lens=txt_seq_lens,
                        attention_kwargs=None,
                        return_dict=False,
                    )[0]
                if not cfg == 1:
                    pred_un = transformer(
                        hidden_states=latents,
                        timestep=timesteps/1000,
                        guidance=None,
                        encoder_hidden_states_mask=batch_prompt_attention_mask_n,
                        encoder_hidden_states=batch_encoder_hidden_states_n,
                        img_shapes=img_shapes,
                        txt_seq_lens=txt_seq_lens,
                        attention_kwargs=None,
                        return_dict=False,
                    )[0]
                    comb_pred = pred_un + 4 * (pred - pred_un)
                    cond_norm = torch.norm(pred, dim=-1, keepdim=True)
                    noise_norm = torch.norm(comb_pred, dim=-1, keepdim=True)
                    noise_pred = comb_pred * (cond_norm / noise_norm)
                    
                else:
                    noise_pred =pred
            dsigma = sigmas_l[t_base] - sigmas_l[t_start]

            latents = latents.to(torch.float32) + dsigma * noise_pred
  
            ############Direct-Ailgn step3 recover image ###############

            latents = (latents-gt_vector)/(1-sigmast)


            vae.enable_tiling()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                latents = unpack_latents(latents, h, w, 8)
                latents = latents.to(vae.dtype)
                latents_mean = (
                    torch.tensor(vae.config.latents_mean)
                    .view(1, vae.config.z_dim, 1, 1, 1)
                    .to(latents.device, latents.dtype)
                )
                latents_std = 1.0 / torch.tensor(vae.config.latents_std).view(1, vae.config.z_dim, 1, 1, 1).to(
                    latents.device, latents.dtype
                )
                latents = latents / latents_std + latents_mean
                image = vae.decode(latents, return_dict=False)[0][:, :, 0]
            image = image.squeeze().unsqueeze(0)  
            image = (image - image.min()) / (image.max() - image.min())
            
            with torch.amp.autocast('cuda'): 

                outputs = (reward_model.SRP_cfg([pos_caption],batch_caption,image,0.00))*(discount[mid_timestep])


            loss = (-outputs)/gradient_accumulation_steps
            loss = loss.mean()
            loss.backward()
    return loss


def gather_tensor(tensor):
    if not dist.is_initialized():
        return tensor
    world_size = dist.get_world_size()
    gathered_tensors = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(gathered_tensors, tensor)
    return torch.cat(gathered_tensors, dim=0)

def train_one_step(
    args,
    device,
    transformer,
    vae,
    reward_model,
    optimizer,
    lr_scheduler,
    prompt_embeds, 
    prompt_attention_masks, 
    prompt_embeds_n, 
    prompt_attention_masks_n, 
    caption, 
    original_length,
    noise_scheduler,
    max_grad_norm,
    mid_timestep,
    step,
    fp,
    visualization_step,
):
    total_loss = 0.0
    optimizer.zero_grad()
    
    loss = SRPO_train(
            args,
            device, 
            transformer,
            vae,
            prompt_embeds, 
            prompt_attention_masks, 
            prompt_embeds_n, 
            prompt_attention_masks_n, 
            original_length,
            reward_model,
            caption,
            mid_timestep,
            step,
            fp,
            visualization_step,

        )
    if visualization_step:
       return 0,0


    avg_loss = loss.detach().clone()
    dist.all_reduce(avg_loss, op=dist.ReduceOp.AVG)
    total_loss += avg_loss.item()
    return total_loss, 0


def main(args):
    torch.backends.cuda.matmul.allow_tf32 = True
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    dist.init_process_group("nccl")
    torch.cuda.set_device(local_rank)
    device = torch.cuda.current_device()
  
    initialize_sequence_parallel_state(args.sp_size)

    if args.seed is not None:
        set_seed(args.seed + rank)

    # Handle the repository creation
    if rank <= 0 and args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)
    supported_models = ["HPS", "CLIP", "PickScore"]
    logger, _ = setup_logger(args.output_dir)
    if rank <= 0:
        log_dir = os.path.join(args.output_dir, "logs")
        tb_writer = SummaryWriter(log_dir=log_dir)
    if args.reward_model == "HPS":
        print(f"Initializing {args.reward_model} reward model...")
        reward_model = HPS().to(device)
    elif args.reward_model == "CLIP":
        print(f"Initializing {args.reward_model} reward model...")
        reward_model = CLIP(is_pickscore=False).to(device)
    elif args.reward_model == "PickScore":
        reward_model = CLIP(is_pickscore=True).to(device)
        print(f"Initializing {args.reward_model} reward model...")
    elif args.reward_model == "AltCLIP":
        reward_model =AltCLIPRM().to(device)
        print(f"Initializing {args.reward_model} reward model...")
    else:
        raise ValueError(
            f"Unsupported reward model: '{args.reward_model}'. "
            f"Please choose from: {supported_models}"
        )
    reward_model.eval()

    main_print(f"--> loading model from {args.pretrained_model_name_or_path}")
        
    from diffusers.models.transformers.transformer_qwenimage import QwenImageTransformer2DModel, QwenImageTransformerBlock
    transformer = QwenImageTransformer2DModel.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="transformer",
            torch_dtype = torch.float32,
    )
    transformer.train()

    if args.resume:

         state_dict = load_file(args.resume)
         transformer.load_state_dict(state_dict)
    fsdp_config = FSDPConfig(
        sharding_strategy="FULL_SHARD",
        backward_prefetch="BACKWARD_PRE",
        cpu_offload=False,  
        num_replicate=1,
        num_shard=world_size,
        mixed_precision_dtype=torch.bfloat16,
        use_device_mesh=False, 
    )
    transformer = fsdp_wrapper(transformer, fsdp_config,)

    ema_handler = None

    apply_fsdp_checkpointing(
            transformer, (QwenImageTransformerBlock), args.selective_checkpointing
        )

    from diffusers.models.autoencoders import AutoencoderKLQwenImage
    vae = AutoencoderKLQwenImage.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
        torch_dtype = torch.bfloat16,
        local_files_only=True,
    ).to(device)
    

    main_print(
        f"--> Initializing FSDP with sharding strategy: {args.fsdp_sharding_startegy}"
    )
    # Load the reference model
    main_print(f"--> model loaded")

    noise_scheduler = None
    reward_model.requires_grad_(False)
    reward_model.eval()
    params_to_optimize = list(transformer.parameters())
    params_to_optimize = list(filter(lambda p: p.requires_grad, params_to_optimize))
    optimizer = torch.optim.AdamW(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=args.weight_decay,
        eps=1e-8,
    )

    init_steps = 0
    main_print(f"optimizer: {optimizer}")

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=1000000,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
        last_epoch=init_steps - 1,
    )

    train_dataset = LatentDataset(args.data_json_path, args.num_latent_t, 0)
    sampler = DistributedSampler(
            train_dataset, rank=rank, num_replicas=world_size, shuffle=False, seed=args.sampler_seed
        )
    
    train_dataloader = DataLoader(
        train_dataset,
        sampler=sampler,
        collate_fn=None,
        pin_memory=True,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
        drop_last=True,
    )

    # Train!
    dist.barrier()
    total_batch_size = (
        world_size
        * args.gradient_accumulation_steps
        / args.sp_size
        * args.train_sp_batch_size
    )
    main_print("***** Running training *****")
    main_print(f"  Num examples = {len(train_dataset)}")
    main_print(f"  Dataloader size = {len(train_dataloader)}")
    main_print(f"  Resume training from step {init_steps}")
    main_print(f"  Instantaneous batch size per device = {args.train_batch_size}")
    main_print(f"  Total optimization steps per epoch = {args.max_train_steps}")
    main_print(
        f"  Total training parameters per FSDP shard = {sum(p.numel() for p in transformer.parameters() if p.requires_grad) / 1e9} B"
    )
    # print dtype
    main_print(f"  Master weight dtype: {transformer.parameters().__next__().dtype}")

    if args.resume_from_checkpoint:
        assert NotImplementedError("resume_from_checkpoint is not supported now.")
    dir_path = './images'
    fp = os.path.join(dir_path, args.image_p)
    os.makedirs(fp, exist_ok=True)
    progress_bar = tqdm(
        range(0, 100000),
        initial=init_steps,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=local_rank > 0,
    )
    step_times = deque(maxlen=100)
    ast=0
    for epoch in range(1000000):
        if isinstance(sampler, DistributedSampler):
            sampler.set_epoch(epoch) # Crucial for distributed shuffling per epoch
        for step, (prompt_embeds, prompt_attention_masks, prompt_embeds_n,prompt_attention_masks_n,caption, original_length) in enumerate(train_dataloader):
            prompt_embeds = prompt_embeds.to(device)
            prompt_attention_masks = prompt_attention_masks.to(device)
            prompt_embeds_n = prompt_embeds_n.to(device)
            prompt_attention_masks_n = prompt_attention_masks_n.to(device)
            start_time = time.time()
            step=ast
            if step == args.checkpointing_steps:
               save_checkpoint_bf16(transformer, rank, args.output_dir,
                                step, epoch)
            mid_timestep_tensor = torch.tensor([random.randint(5, args.timestep_length-5)], device=device)
            max_grad_norm = args.max_grad_norm
            visualization_step = False
            loss, grad_norm = train_one_step(
                args,
                device, 
                transformer,
                vae,
                reward_model,
                optimizer,
                lr_scheduler,
                prompt_embeds, 
                prompt_attention_masks, 
                prompt_embeds_n, 
                prompt_attention_masks_n, 
                caption, 
                original_length,
                noise_scheduler,
                max_grad_norm,
                mid_timestep_tensor,
                step,
                fp,
                visualization_step,
            )
            grad_norm = transformer.clip_grad_norm_(max_grad_norm)
            grad_norm = torch.tensor(grad_norm, device=device)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            ast+=1
            loss_type = 'rm_loss'
            step_time = time.time() - start_time
            step_times.append(step_time)

            progress_bar.update(1)
            if rank <= 0 and not visualization_step:
                progress_info = {
                loss_type: f"{loss:.4f}",
                "step_time": f"{step_time:.2f}s",
                "grad_norm": grad_norm,
                }
                logger.info(f"Progress: {step}/{args.max_train_steps} | Details: {progress_info}")


    if get_sequence_parallel_state():
        destroy_sequence_parallel_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # dataset & dataloader
    parser.add_argument("--data_json_path", type=str, required=True)
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=10,
        help="Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process.",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=16,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--num_latent_t",
        type=int,
        default=1,
        help="number of latent frames",
    )
    # text encoder & vae & diffusion model
    parser.add_argument("--image_p", type=str)
    parser.add_argument("--pretrained_model_name_or_path", type=str)
    parser.add_argument("--dit_model_name_or_path", type=str, default=None)
    parser.add_argument("--vae_model_path", type=str, default=None, help="vae model.")
    parser.add_argument("--cache_dir", type=str, default="./cache_dir")

    # diffusion setting

    parser.add_argument("--ema_decay", type=float, default=0.995)
    parser.add_argument("--ema_start_step", type=int, default=0)
    parser.add_argument("--train_cfg", type=float, default=1.0)
    parser.add_argument(
        "--precondition_outputs",
        action="store_true",
        help="Whether to precondition the outputs of the model.",
    )

    # validation & logs
    parser.add_argument(
        "--seed", type=int, default=None, help="A seed for reproducible training."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=50,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints can be used both as final"
            " checkpoints in case they are better than the last checkpoint, and are also suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )

    # optimizer & scheduler & Training
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=8,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--lr_warmup_steps",
        type=int,
        default=10,
        help="Number of steps for the warmup in the lr scheduler.",
    )
    parser.add_argument(
        "--max_grad_norm", default=2.0, type=float, help="Max gradient norm."
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument("--selective_checkpointing", type=float, default=1.0)
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--use_cpu_offload",
        action="store_true",
        help="Whether to use CPU offload for param & gradient & optimizer states.",
    )

    parser.add_argument("--sp_size", type=int, default=1, help="For sequence parallel")
    parser.add_argument(
        "--train_sp_batch_size",
        type=int,
        default=1,
        help="Batch size for sequence parallel training",
    )

    parser.add_argument("--fsdp_sharding_startegy", default="full")

    # lr_scheduler
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant_with_warmup",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of cycles in the learning rate scheduler.",
    )
    parser.add_argument(
        "--lr_power",
        type=float,
        default=1.0,
        help="Power factor of the polynomial scheduler.",
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.01, help="Weight decay to apply."
    )
    parser.add_argument(
        "--master_weight_type",
        type=str,
        default="fp32",
        help="Weight type to use - fp32 or bf16.",
    )

    #GRPO training
    parser.add_argument(
        "--h",
        type=int,
        default=None,   
        help="video height",
    )
    parser.add_argument(
        "--w",
        type=int,
        default=None,   
        help="video width",
    )
    parser.add_argument(
        "--t",
        type=int,
        default=None,   
        help="video length",
    )
    parser.add_argument(
        "--sampling_steps",
        type=int,
        default=None,   
        help="sampling steps",
    )
    parser.add_argument(
        "--eta",
        type=float,
        default=None,   
        help="noise eta",
    )
    parser.add_argument(
        "--sampler_seed",
        type=int,
        default=None,   
        help="seed of sampler",
    )
    parser.add_argument(
        "--loss_coef",
        type=float,
        default=1.0,   
        help="the global loss should be divided by",
    )
    parser.add_argument(
        "--ignore_last",
        action="store_true",
        default=False,
        help="whether ignore last step of mdp",
    )
    parser.add_argument(
        "--shift",
        type = float,
        default=1.0,
        help="shift for timestep scheduler",
    )
    parser.add_argument(
        "--timestep_length",
        type=int,
        default=100,   
        help="number of timestep",
    )
    parser.add_argument(
        "--groundtruth_ratio",
        type=float,
        default=0.9,   
        help="groundtruth_ratio used for recovering image",
    )
    parser.add_argument(
        '--train_timestep',
        type=int,        
        nargs=2,          
        default=[5, 25],
        help='timestep intervals for training'
    )

    parser.add_argument(
        '--discount_pos',
        type=float,        
        nargs=2,          
        default=[0.1, 0.25],
        help='later-time discount for postive branch'
    )
    parser.add_argument(
        '--discount_inv',
        type=float,        
        nargs=2,        
        default=[0.3, 0.01],
        help='later-time discount for inversion branch'
    )
    parser.add_argument(
        '--vis_cfg',
        type=int,        
        default=3.5,
        help='cfg for visualization'
    )
    parser.add_argument(
        '--vis_sampling_step',
        type=int,        
        default=50,
        help='visualization_sampling_step'
    )
    parser.add_argument(
        '--vis_size',
        type=int,        
        default=1024,
        help='visualization_image_size'
    )
    parser.add_argument(
        '--offline_end',
        type=int,        
        default=30,
        help='visualization_image_size'
    )
    parser.add_argument(
        '--reward_model',
        type=str,        
        default='HPS',
        help='rm'
    )
    parser.add_argument(
        '--resume',
        type=str,        
        default=None,
        help='rm'
    )
    parser.add_argument(
        '--offline_dict',
        type=str,        
        default=None,
        help='pkl for path and'
    )


    args = parser.parse_args()
    main(args)
