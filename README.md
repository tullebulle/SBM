# SBM: Scalable Beckmann Maps

<p align="center"><img src="assets/teaser.jpg" width="100%"></p>
<p align="center"><em>SBM-XL text-to-image samples: one network evaluation per image, 256&times;256.</em></p>

One-step generators trained with self-guided Beckmann transport: class-conditional ImageNet 256&times;256 (FID 1.61),
text-to-image (GenEval 0.856) and action-conditioned world models (Bridge-V2, RT-1), all with a single network
evaluation.

## Install

```bash
git clone https://github.com/tullebulle/SBM.git && cd SBM
uv sync                  # Python 3.12, PyTorch 2.9 (CUDA 12.8); `uv sync --extra wm` for the world models
source .venv/bin/activate
```

## Checkpoints

The checkpoints are on Hugging Face ([sbm-models/sbm](https://huggingface.co/sbm-models/sbm)); request access, then
`hf auth login`. One command downloads them with the frozen networks they need:

```bash
bash download.sh              # everything, or one of: imagenet | t2i | wm
```

| task | model | `--model` | result |
|---|---|---|---|
| ImageNet 256 | SBM-XL, DiT-XL/1-DH in the RAE latent | `rae-xl` | FID 1.61 |
| | SBM-B, DiT-B/1-DH in the RAE latent | `rae-b` | FID 2.79 |
| | SBM-XL/2, DiT-XL/2 in the SD-VAE latent | `sdvae-xl` | FID 3.56 |
| | SBM-B/2, DiT-B/2 in the SD-VAE latent | `sdvae-b` | FID 5.83 |
| Text-to-image 256 | SBM-XL (815M) | `xl` | GenEval 0.856 |
| | SBM-B (254M) | `b` | GenEval 0.833 |
| World model | Bridge-V2 | `bridge` | FVD 18.1 |
| | Bridge-V2, with the DINOv2 feature term | `bridge-dino` | FVD 21.7 |
| | RT-1 | `rt1` | FVD 16.2 |
| | RT-1, with the DINOv2 feature term | `rt1-dino` | FVD 19.7 |

FID-50k against the ADM ImageNet reference (SBM-XL in the RAE latent reads 1.61 on an A100 and 1.65 on H100 / H200);
GenEval at sampling seed 1; world models on the full validation split (DriftWorld's 8-frame protocol).

## Sample

```bash
python sample.py t2i "a red fox curled up asleep in fresh snow, soft morning light"     # -> samples/t2i/
python sample.py imagenet --classes 88 207 279 360 387 974                             # -> samples/imagenet/
python sample.py wm --model bridge                                                     # -> samples/wm_bridge/*.mp4
```

Each checkpoint carries the settings of its published numbers; `--model`, `--seed`, `--per-prompt` / `--per-class`,
`--omega` and `--label-scale` change them. The world-model rollouts run on example episodes downloaded with the
checkpoints (ground truth left, SBM right). In Python:

```python
from sbm_t2i.infer import load_model, generate
model, cfg, _ = load_model("store/ckpt/t2i/sbm_t2i_xl_step1200000_ema0_9999.pt")
images = generate(model.cuda(), cfg, ["a lighthouse on a cliff at dusk"], seed=0)    # uint8 (1, 3, 256, 256)
```

## Evaluate

```bash
# ImageNet FID-50k (RAE latent), one GPU or torchrun; the reference statistics come with the checkpoints
python scripts/rae/eval_fid.py checkpoint_path=store/ckpt/imagenet/sbm_rae_xl_step1042500_ema0_9999.pt
# ImageNet FID-50k (SD-VAE latent)
python -m sbm.cli eval store/ckpt/imagenet/sbm_sdvae_xl_step355000_ema0_9999.pt
# GenEval: samples, then the scorer (its environment: bash scripts/t2i/geneval/build_env.sh && bash scripts/t2i/geneval/build_mmcv.sh)
python scripts/t2i/geneval_sample.py ckpt=store/ckpt/t2i/sbm_t2i_xl_step1200000_ema0_9999.pt out=geneval/xl seed=1
bash scripts/t2i/geneval/score.sh geneval/xl geneval/xl_score
# World models: every validation episode on all visible GPUs (needs the data below)
bash scripts/rae/wm/eval.sh store/ckpt/wm/sbm_wm_bridge_step660000_ema0_999.pt wm_eval/bridge
```

## Train

Data goes in `store/` (or `$SBM_STORE`): ImageNet as `store/data/imagenet_1k/{train,val}/<wnid>/*.JPEG`; the other
datasets are prepared by the scripts below.

```bash
# ImageNet, RAE latent: fine-tune the released SBM-XL (images are encoded on the fly)
python scripts/rae/train.py +experiment=rae_record init_weights_from=store/ckpt/imagenet/sbm_rae_xl_step1042500_ema0_9999.pt \
    loss.guided_cbm.guide.autoguide.enabled=false trainer.devices=8
# ImageNet, SD-VAE latent: encode ImageNet once (one process per GPU with --shard i --num-shards n), then train
python scripts/sbm/encode_imagenet.py store/data/imagenet_1k/train store/data/imagenet_latents_packed_adm
python -m sbm.cli train b2_sbm name=b2                          # SBM-B/2 from scratch, all visible GPUs
python -m sbm.cli train xl_finetune name=xl_ft                  # fine-tune the released SBM-XL/2
# Text-to-image: CC12M with recaptions and the 120k alignment mix (scripts/t2i/data), then
python -m sbm_t2i.cli train t2i_rae_xl1dh                       # or t2i_rae_b1dh_t15; all visible GPUs
# World models: data (download, conversion, SD3-VAE latents; ~300 GB per dataset), then training
bash scripts/rae/wm/data/fetch_driftworld.sh && bash scripts/rae/wm/data/setup_convert_env.sh
bash scripts/rae/wm/data/prepare.sh bridge                      # or rt1
python scripts/rae/train.py +experiment=wm_bridge trainer.devices=2    # wm_rt1: trainer.devices=4
```

The recipes and the cool-down schedules of the released models are in the headers of `configs/rae/experiment/*.yaml`,
`configs/sbm/exp/*.yaml` and `configs/t2i/exp/*.yaml`. Fine-tuning any released checkpoint: `init_weights_from=<file>`
(RAE line and world models), `init.from=<file>` (SD-VAE line and text-to-image).

## Tests

```bash
python -m pytest tests/sbm && python -m pytest tests/rae && python -m pytest tests/t2i    # CPU; run them separately
```

## License

MIT (`LICENSE`). `third_party/RAE` keeps its MIT license; the DiT backbones derive from DiT (Meta) and keep its copyright
headers; the flash-attention JVP kernels build on FlashAttention (BSD-3-Clause). The checkpoints were trained on ImageNet,
CC12M and its recaptions, BLIP3o-60k, the DALL-E 3 dataset, ShareGPT-4o-Image, BridgeData V2 and RT-1; their use is subject
to those datasets' terms. The world models work in the latent of the SD3 VAE (Stability AI Non-Commercial Research
Community License). DriftWorld's data converter is fetched from its repository, which carries no license.
