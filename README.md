# udl_assignment_2

CIFAR-10 image generation assignment (UDL) — deep generative models implemented in PyTorch
and run on Google Colab.

## Contents

| File | Description |
|---|---|
| [`Part_A_VAE_VQVAE_merged.ipynb`](Part_A_VAE_VQVAE_merged.ipynb) | **Part A**, merged/final version. (β-)VAE for β ∈ {1, 2, 4, 10} and VQ-VAE + PixelCNN prior for codebook size K ∈ {512, 256, 128}. Self-contained: CIFAR-10 auto-downloads via torchvision, FID via `torch-fidelity`. Runnable end-to-end on a free Colab T4. |
| `Part_A_VAE_VQVAE.ipynb` | Part A, earlier draft (kept for reference). |
| `part_a_vae_vqvae_1.ipynb` | Part A, earlier draft (kept for reference). |
| `Part_B_WGAN.ipynb` | **Part B**. DCGAN-style generator/critic trained two ways: WGAN (weight clipping) and WGAN-GP (gradient penalty), with FID comparison. |
| `part_b_wgan_1.ipynb` | Part B, earlier draft (kept for reference). |
| `assignment 2 (1).pdf` / `.docx` | Assignment brief. |

## What's implemented

**Part A — VAE / VQ-VAE**
- β-VAE: PSNR vs. epoch and FID (sampled from the prior) across β
- VQ-VAE: reconstruction PSNR, codebook perplexity/usage, and reconstruction FID across K
- PixelCNN prior trained over the VQ-VAE discrete latents (per K) to sample new latent grids and decode them into images
- Comparison plots/tables across β and K

**Part B — WGAN / WGAN-GP**
- Shared DCGAN-style generator/critic architecture
- WGAN with weight clipping (RMSProp, per the original paper)
- WGAN-GP with gradient penalty (Adam, per Gulrajani et al.)
- Training curves and FID comparison between the two objectives

## Running

Open a notebook in Google Colab, set `Runtime > Change runtime type > GPU`, then `Runtime > Run all`.
Dependencies (`torch-fidelity`) and the CIFAR-10 dataset are fetched automatically — no manual
upload or Drive mount required for the merged Part A notebook.
