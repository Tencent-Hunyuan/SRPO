<div align=“center” style=“font-family: charter;”>
<h1 align="center">Directly Aligning the Full Diffusion Trajectory with Fine-Grained Human Preference </h1>
<div align="center">
  <a href='https://arxiv.org/abs/2509.06942'><img src='https://img.shields.io/badge/ArXiv-red?logo=arxiv'></a>  &nbsp;
  <a href='https://huggingface.co/tencent/SRPO/'><img src='https://img.shields.io/badge/Flux-blue?logo=huggingface'></a> &nbsp;
  <a href='https://tencent.github.io/srpo-project-page/'><img src='https://img.shields.io/badge/%F0%9F%92%BB_Project-SRPO-blue'></a> &nbsp;
</div>
<div align="center">
  Xiangwei Shen<sup>1,2,3*</sup>,
  <a href="https://scholar.google.com/citations?user=Lnr1FQEAAAAJ&hl=zh-CN" target="_blank"><b>Zhimin Li</b></a><sup>1*</sup>,
  <a href="https://scholar.google.com.hk/citations?user=Fz3X5FwAAAAJ" target="_blank"><b>Zhantao Yang</b></a><sup>1</sup>,
  <a href="https://shiyi-zh0408.github.io/" target="_blank"><b>Shiyi Zhang</b></a><sup>3</sup>,
  Yingfang Zhang<sup>1</sup>,
  Donghao Li<sup>1</sup>,
  <br>
  <a href="https://scholar.google.com/citations?user=VXQV5xwAAAAJ&hl=en" target="_blank"><b>Chunyu Wang</b></a><sup>1✝</sup>,
  <a href="https://openreview.net/profile?id=%7EQinglin_Lu2" target="_blank"><b>Qinglin Lu</b></a><sup>1</sup>,
  <a href="https://andytang15.github.io" target="_blank"><b>Yansong Tang</b></a><sup>3,✉️</sup>
</div>
<div align="center">
  <sup>1</sup>Hunyuan, Tencent
  <br>
  <sup>2</sup>School of Science and Engineering, The Chinese University of Hong Kong, Shenzhen
  <br>
  <sup>3</sup>Shenzhen International Graduate School, Tsinghua University
  <br>
  <sup>*</sup>Equal contribution
  <sup>✝</sup>Project lead
  <sup>✉️</sup>Corresponding author
</div>

![head](assets/head.jpg)

## 🎉 Key Features
1. **Direct Align**: We introduce a new sampling strategy for diffusion fine-tuning that can effectively restore highly noisy images, leading to an optimization process that is more stable and less computationally demanding, especially during the initial timesteps.
2. **Faster Training**:   By rolling out only a single image and optimizing directly with analytical gradients—a key distinction from GRPO—our method achieves significant performance improvements for FLUX.1.dev in under 10 minutes of training. To further accelerate the process, our method supports replacing online rollouts entirely with a small dataset of real images; we find that fewer than 1500 images are sufficient to effectively train FLUX.1.dev.
3. **Free of Reward Hacking**: We have improved the training strategy for method that direct backpropagation on reward signal (such as ReFL and DRaFT). Moreover, we directly regularize the model using negative rewards, without the need for KL divergence or a separate reward system. In our experiments, this approach achieves comparable performance with multiple different rewards, improving the perceptual quality of FLUX.1.dev without suffering from reward hacking issues, such as overfitting to color or oversaturation preferences.
4. **Potential for Controllable Fine-tuning**: For the first time in online RL, we incorporate dynamically controllable text conditions, enabling on-the-fly adjustment of reward preference towards styles within the scope of the reward model.

## 🔥 News
- __[2026.2.24]__:  🎉 We released the training code for online/offline training by SRPO, built upon the Qwen-Image.
- __[2025.10.26]__: 👑 We achieved the Top1 on [Artificial Analysis Leaderboard](https://artificialanalysis.ai/image/leaderboard/text-to-image?open-weights=true) for text-to-image open-source models.
  
  <img width="701" height="213" alt="image" src="https://github.com/user-attachments/assets/bc8765c3-5a1e-4e66-89f8-368136ec8492" />
- __[2025.9.12]__:  🎉 We released the complete training code. We also share tips and experiences to help you train your models. You’re welcome to discuss and ask questions in the issues! 💬✨
- __[2025.9.12]__:  🎉 We provide a standard workflow—feel free to use it in ComfyUI.
- __[2025.9.8]__:   🎉 We released the paper, checkpoint, inference code.


## 📑 Open-Source Plan

- [x] Training code (FLUX.1-dev)
- [x] Support Qwen-Image
- [ ] Extend support to other models (Qwen-Image-Edit, FLUX 2 etc.)

## 🛠️ Dependencies and Installation

```bash
conda create -n SRPO python=3.10.16 -y
conda activate SRPO
bash ./env_setup.sh
```

> 💡 The environment dependencies are essentially the same as [DanceGRPO](https://github.com/XueZeyue/DanceGRPO).

## 🤗 Download Models

### Model Cards

| Model | Hugging Face Download URL |
| :---: | :-----------------------: |
| SRPO  | [diffusion_pytorch_model](https://huggingface.co/tencent/SRPO/tree/main) |

### Download SRPO Checkpoint

Download `diffusion_pytorch_model.safetensors` from [Hugging Face](https://huggingface.co/tencent/SRPO):

```bash
mkdir -p ./srpo
huggingface-cli login
huggingface-cli download --resume-download Tencent/SRPO diffusion_pytorch_model.safetensors --local-dir ./srpo/
```

### Download Base Model

Load your FLUX cache or download [black-forest-labs/FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev):

```bash
mkdir -p ./data/flux
huggingface-cli login
huggingface-cli download --resume-download black-forest-labs/FLUX.1-dev --local-dir ./data/flux
```

## 🔑 Inference

### Using ComfyUI

You can use SRPO in [ComfyUI](https://github.com/comfyanonymous/ComfyUI).

Load the following image in ComfyUI to get the workflow, or load the JSON file directly: [SRPO-workflow](comfyui/SRPO-workflow.json).

> **Tip:** The workflow JSON info is embedded in the image file.

![Example](comfyui/SRPO-workflow.png)

### Quick Start

```python
import torch
from diffusers import FluxPipeline
from safetensors.torch import load_file

prompt='The Death of Ophelia by John Everett Millais, Pre-Raphaelite painting, Ophelia floating in a river surrounded by flowers, detailed natural elements, melancholic and tragic atmosphere'
pipe = FluxPipeline.from_pretrained('./data/flux',
        torch_dtype=torch.bfloat16,
        use_safetensors=True
    ).to("cuda")
state_dict = load_file("./srpo/diffusion_pytorch_model.safetensors")
pipe.transformer.load_state_dict(state_dict)
image = pipe(
    prompt,
    guidance_scale=3.5,
    height=1024,
    width=1024,
    num_inference_steps=50,
    max_sequence_length=512,
    generator=generator
).images[0]
```

### Batch Inference

Run inference with our example cases. Replace `model_path` in `vis.py` before running:

```bash
torchrun --nnodes=1 --nproc_per_node=8 \
    --node_rank 0 \
    --rdzv_endpoint $CHIEF_IP:29502 \
    --rdzv_id 456 \
    vis.py
```

## 🚚 Training

### Prepare Training Model
1. Pretrain Model: download the FLUX.dev.1 checkpoints from [huggingface](https://huggingface.co/black-forest-labs/FLUX.1-dev) to `./data/flux`.
```bash
mkdir data
mkdir ./data/flux
huggingface-cli login
huggingface-cli download --resume-download  black-forest-labs/FLUX.1-dev --local-dir ./data/flux
```
2. Reward Model: download the HPS-v2.1(HPS_v2.1_compressed.pt) and CLIP H-14 checkpoints from [huggingface](https://huggingface.co/xswu/HPSv2/tree/main) to `./hps_ckpt`.
```bash
mkdir ./data/hps_ckpt
huggingface-cli login
huggingface-cli download --resume-download xswu/HPSv2 HPS_v2.1_compressed.pt --local-dir ./data/hps_ckpt
huggingface-cli download --resume-download laion/CLIP-ViT-H-14-laion2B-s32B-b79K open_clip_pytorch_model.bin --local-dir ./data/hps_ckpt
```
3. (Optional) Reward Model: download the PickScore checkpoint from [huggingface](https://huggingface.co/yuvalkirstain/PickScore_v1) to `./data/ps`.
```bash
mkdir ./data/ps
huggingface-cli login
python ./scripts/huggingface/download_hf.py --repo_id yuvalkirstain/PickScore_v1  --local_dir ./data/ps
python ./scripts/huggingface/download_hf.py --repo_id laion/CLIP-ViT-H-14-laion2B-s32B-b79K --local_dir ./data/clip
```
### Prepare Training Data

#### Pre-Extract Text Embeddings

For online version, no image-text pairs are needed — only text prompts.

```bash
# Write training prompts into ./prompts.txt
vim ./prompts.txt

# Pre-extract text embeddings from your custom training dataset to boost training efficiency
bash scripts/preprocess/preprocess_flux_rl_embeddings.sh

# Copy your caption file to the embeddings directory
cp videos2caption2.json ./data/rl_embeddings
```

For Qwen-Image:

```bash
bash scripts/preprocess/preprocess_qwen_rl_embeddings.sh
```

#### (Optional) Prepare Offline Data

Prepare an offline `.pkl` file that maps captions to image paths:

```python
img_path = offline_dict["An image of a cat"]
```

Bucket the data so that samples in the same batch share the same resolution (otherwise the program will hang):

```bash
python prepare_prompt.py \
    --input_json <../data/rl_embeddings/videos2caption.json> \
    --output_json <output_path.json> \
    --batch_size 64
```

### Full-Parameter Training

#### FLUX

- **HPS-v2.1** as Reward Model:

    ```bash
    bash scripts/finetune/SRPO_training_hpsv2.sh
    ```

- **(Optional) PickScore** as Reward Model:

    ```bash
    bash scripts/finetune/SRPO_training_ps.sh
    ```

    > ⚠️ The current control words are designed for HPS-v2.1, so training with PickScore may yield suboptimal results compared to HPS due to this mismatch.

- **Distributed training** with `pdsh`:

    ```bash
    #!/bin/bash
    echo "$NODE_IP_LIST" | tr ',' '\n' | sed 's/:8$//' | grep -v '1.1.1.1' > /tmp/pssh.hosts
    node_ip=$(paste -sd, /tmp/pssh.hosts)
    pdsh -w $node_ip "conda activate SRPO; cd <project_path>; bash scripts/finetune/SRPO_training_hpsv2.sh"
    ```

#### Qwen-Image

- **HPS-v2.1** as Reward Model:

    ```bash
    bash scripts/finetune/SRPO_training_qwen.sh
    ```

- **Offline Training Parameters:**

    | Parameter | Type | Description |
    | :--- | :--- | :--- |
    | `--offline_end` | `int` | Training step at which to switch from offline to online mode. Default: `-1` (disables the switch; runs in online mode only). |
    | `--offline_dict` | `str` | Path to the offline data `.pkl` file. Must be a dictionary mapping captions to image paths: `{caption: image_path}`. |

### How to Support Custom Models

1. Modify `preprocess_flux_embedding.py` and `latent_flux_rl_datasets.py` to pre-extract text embeddings from your custom training dataset — this boosts training efficiency.
2. Adjust `args.vis_sampling_step` to modify the sigma schedule. Typically, this value should match the model's regular inference steps.
3. Direct propagation requires significant GPU memory. Enabling VAE gradient checkpointing before reward calculation reduces memory usage greatly.
4. If implementing outside FastVideo, first disable the inversion branch to check for reward hacking — its presence likely indicates a correct implementation.
5. Pure Direct-Align works for SRPO-unsupported tasks (e.g., OCR, image editing) with minimal code changes.

### Hyperparameter Recommendations

For best results, use these settings as a starting point and adjust for your model/dataset:

1. **`batch_size`**: Larger sizes generally improve quality. For FLUX.1-dev reinforcement learning under current settings, 32 works well.
2. **`learning_rate`**: Values in the range of 1e-5 to 1e-6 work for most models.
3. **`train_timestep`**: Focus on early-to-middle diffusion stages. Too early (e.g., `sigma > 0.99`) causes structural distortions; too late encourages color-based reward hacking.
4. **`discount_inv`** & **`discount_denoise`**: Let `discount_inv = [a, b]`, `discount_denoise = [c, d]`. To preserve structure, set `c` slightly greater than `b` (avoids early layout corruption). To fix color oversaturation, set `a` slightly greater than `d` (tempers aggressive tones). The current hyperparameters work well for most in-house models and serve as a good baseline.

## 🎉 Acknowledgement

We referenced the following works, and appreciate their contributions to the community:

- [FastVideo](https://github.com/hao-ai-lab/FastVideo)
- [DanceGRPO](https://github.com/XueZeyue/DanceGRPO)

## 🔗 BibTeX

If you find SRPO useful for your research and applications, please cite using this BibTeX:

```bibtex
@misc{shen2025directlyaligningdiffusiontrajectory,
      title={Directly Aligning the Full Diffusion Trajectory with Fine-Grained Human Preference},
      author={Xiangwei Shen and Zhimin Li and Zhantao Yang and Shiyi Zhang and Yingfang Zhang and Donghao Li and Chunyu Wang and Qinglin Lu and Yansong Tang},
      year={2025},
      eprint={2509.06942},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2509.06942},
}
```

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=Tencent-Hunyuan/SRPO&type=Date)](https://www.star-history.com/#Tencent-Hunyuan/SRPO&Date)
