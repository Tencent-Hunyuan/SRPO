# Copyright (c) [2025] [FastVideo Team]
# Copyright (c) [2025] [ByteDance Ltd. and/or its affiliates.]
# SPDX-License-Identifier: [Apache License 2.0] 
#
# This file has been modified by [ByteDance Ltd. and/or its affiliates.] in 2025.
#
# Original file was released under [Apache License 2.0], with the full license text
# available at [https://github.com/hao-ai-lab/FastVideo/blob/main/LICENSE].
#
# This modified file is released under the same license.

import torch
from torch.utils.data import Dataset
import json
import os
import random


class LatentDataset(Dataset):
    def __init__(
        self, json_path, num_latent_t, cfg_rate,
    ):
        # data_merge_path: video_dir, latent_dir, prompt_embed_dir, json_path
        self.json_path = json_path
        self.cfg_rate = cfg_rate
        self.datase_dir_path = os.path.dirname(json_path)
        #self.video_dir = os.path.join(self.datase_dir_path, "video")
        #self.latent_dir = os.path.join(self.datase_dir_path, "latent")
        self.prompt_embed_dir = os.path.join(self.datase_dir_path, "prompt_embed")
        self.prompt_attention_mask_dir = os.path.join(
            self.datase_dir_path, "prompt_attention_mask"
        )
        self.prompt_embed_dir_neg = os.path.join(self.datase_dir_path, "prompt_embed_neg")
        self.prompt_attention_mask_dir_neg = os.path.join(
            self.datase_dir_path, "prompt_attention_mask_neg"
        )
        with open(self.json_path, "r") as f:
            self.data_anno = json.load(f)
        # json.load(f) already keeps the order
        # self.data_anno = sorted(self.data_anno, key=lambda x: x['latent_path'])
        self.num_latent_t = num_latent_t
        # just zero embeddings [256, 4096]
        self.uncond_prompt_embed = torch.zeros(256, 4096).to(torch.float32)
        # 256 zeros
        self.uncond_prompt_mask = torch.zeros(256).bool()
        self.lengths = [
            data_item["length"] if "length" in data_item else 1
            for data_item in self.data_anno
        ]

    def __getitem__(self, idx):
        #latent_file = self.data_anno[idx]["latent_path"]
        prompt_embed_file = self.data_anno[idx]["prompt_embed_path"]
        prompt_attention_mask_file = self.data_anno[idx]["prompt_attention_mask"]
        prompt_embed_file_neg = self.data_anno[idx]["prompt_embed_path_neg"]
        prompt_attention_mask_file_neg = self.data_anno[idx]["prompt_attention_mask_neg"]
        prompt_embed = torch.load(
            os.path.join(self.prompt_embed_dir, prompt_embed_file),
            map_location="cpu",
            weights_only=True,
        )
        prompt_attention_mask = torch.load(
            os.path.join(
                self.prompt_attention_mask_dir, prompt_attention_mask_file
            ),
            map_location="cpu",
            weights_only=True,
        )
        prompt_embed_neg = torch.load(
            os.path.join(self.prompt_embed_dir_neg, prompt_embed_file_neg),
            map_location="cpu",
            weights_only=True,
        )
        prompt_attention_mask_neg = torch.load(
            os.path.join(
                self.prompt_attention_mask_dir_neg, prompt_attention_mask_file_neg
            ),
            map_location="cpu",
            weights_only=True,
        )
        return prompt_embed, prompt_attention_mask,prompt_embed_neg,prompt_attention_mask_neg,self.data_anno[idx]['caption'], self.data_anno[idx]['original_length']

    def __len__(self):
        return len(self.data_anno)


def latent_collate_function(batch):
    # return latent, prompt, latent_attn_mask, text_attn_mask
    # latent_attn_mask: # b t h w
    # text_attn_mask: b 1 l
    # needs to check if the latent/prompt' size and apply padding & attn mask
    prompt_embeds, prompt_attention_masks,pe_n,pa_n,caption, original_length = zip(*batch)
    # attn mask
    prompt_embeds = torch.stack(prompt_embeds, dim=0)
    prompt_attention_masks = torch.stack(prompt_attention_masks, dim=0)
    #latents = torch.stack(latents, dim=0)
    return prompt_embeds, prompt_attention_masks, pe_n,pa_n,caption, original_length