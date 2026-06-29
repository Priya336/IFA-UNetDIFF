# IFA-UNetDiff: Inter-Feature Attention Guided Diffusion for Unsupervised Hyperspectral Pansharpening

**Priya Goswami, Gautam Kumar** — NIT Delhi

A lightweight diffusion-based framework for unsupervised hyperspectral pansharpening. Integrates an Inter-Feature Attention (IFA) module into the U-Net decoder to selectively transfer spatial details from the PAN image into hyperspectral features during reconstruction.

---

## Requirements

```bash
pip install torch torchvision numpy scipy scikit-image matplotlib h5py
```

---

## Data

Place `.mat` files under `./data/`. Each file must contain:
- `HRMS` — high-resolution hyperspectral (ground truth)
- `LRMS` — low-resolution hyperspectral input
- `PAN` — panchromatic image

Datasets used: [Pavia University](https://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes) · [Houston](https://hyperspectral.ee.uh.edu/) · [Chikusei](https://naotoyokoya.com/Download.html)

---

## Pretrained Model

Download pretrained diffusion weights from [PLRDiff](https://github.com/xyrui/PLRDiff) and pass the path via `--resume_state`.

---

## Inference

```bash
python demo_syn2.py \
  -c base.json \
  -gpu 0 \
  -dn Chikusei \
  -sr ./results \
  -rs /path/to/pretrained/I190000_E97 \
  -rank 3 -scale 4 -ks 11 \
  -step 2000 -accstep 500 \
  -krtype 0
```

| Argument | Description | Default |
|---|---|---|
| `-dn` | Dataset name (matches `./data/<name>.mat`) | `Chikusei` |
| `-rs` | Pretrained model path (without `_gen.pth`) | — |
| `-rank` | Spectral subspace dimension | `3` |
| `-scale` | Spatial downsampling factor | `4` |
| `-krtype` | `0` = estimate kernel/SRF, `1` = load from `./estKR/` | `0` |
| `-res` | Residual mode: `no` or `opt` | `no` |

---

## Results

| Dataset | PSNR ↑ | SSIM ↑ | SCC ↑ | Q2n ↑ |
|---|---|---|---|---|
| Chikusei (Base) | 30.73 | 0.8160 | 0.8115 | 0.6164 |
| Chikusei (Ours) | 30.63 | **0.8440** | **0.9311** | **0.9295** |
| Pavia (Base) | 32.14 | 0.9392 | 0.8397 | 0.4573 |
| Pavia (Ours) | **32.19** | 0.8864 | **0.9410** | **0.9395** |
| Houston (Base) | **37.69** | 0.9029 | 0.9155 | 0.8874 |
| Houston (Ours) | 37.68 | **0.9208** | **0.9268** | **0.9256** |

---

## Citation

```bibtex
@article{goswami2025ifaunetdiff,
  title   = {IFA-UNetDiff: An Inter-Feature Attention Guided Diffusion Framework for Unsupervised Hyperspectral Pansharpening},
  author  = {Goswami, Priya and Kumar, Gautam},
  journal = {Elsevier},
  year    = {2025}
}
```

Backbone adapted from [PLRDiff](https://github.com/xyrui/PLRDiff) · Bicubic resampling from [bicubic_pytorch](https://github.com/sanghyun-son/bicubic_pytorch)
