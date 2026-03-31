# AGFT: Alignment-Guided Fine-Tuning for Zero-Shot Adversarial Robustness of Vision-Language Models (CVPR 2026)

<p align="center">
  <p align="center" margin-bottom="0px">
    <strong>Yubo Cui</strong></a>
    ·
    <strong>Xianchao Guan</strong></a>
    ·
    <strong>Zijun Xiong</strong></a>
    ·
    <strong>Zheng Zhang</strong></a>
    <p align="center" margin-top="0px"><a href="https://arxiv.org">https://arxiv.org</a></p>
</p>

## Environment setup

### install torch

`pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu126`

### install clip

`pip install ftfy regex tqdm`

`pip install git+https://github.com/openai/CLIP.git`

### install other packages

`pip install -r requirements.txt`

## Datasets

The preparation of the dataset can refer to `torchvision.datasets`, with the default directory being `../dataset`, which can be modified through parameter `--root`.

## Run

### train AGFT

`CUDA_VISIBLE_DEVICES='1,0' OMP_NUM_THREADS=1 torchrun --nproc_per_node 2 main.py --method='AGFT' --use_float --enable_amp --time='40_180_4' --train_or_val --method_args='{"AGFT": {"logit_scale": 180,"scale_ratio": 0.4}}' --lr=4e-4`

### validate on zero-shot datasets

`CUDA_VISIBLE_DEVICES='1,0' OMP_NUM_THREADS=1 torchrun --nproc_per_node 2 main.py --method='AGFT' --use_float --enable_amp --weight_path='40_180_4_ViT-B_32_AGFT_ImageNet_10.pth' --val_attack='PGD'`

### train TeCoA

`CUDA_VISIBLE_DEVICES='1,0' OMP_NUM_THREADS=1 torchrun --nproc_per_node 2 main.py --method='TeCoA' --use_float --enable_amp --time='1' --train_or_val --lr=1e-5`

## If you find this useful in your research, please cite this work:
```
@inproceedings{cui2026agft,
  author = {Cui, Yubo and Guan, Xianchao and Xiong, Zijun and Zhang, Zheng},
  title = {AGFT: Alignment-Guided Fine-Tuning for Zero-Shot Adversarial Robustness of Vision-Language Models},
  booktitle = {IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year = {2026}
}
```
