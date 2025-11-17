# Project Overview

This project explores paired image-to-image translation using a Pix2Pix-based conditional GAN.
The primary objective is to learn a direct mapping between two biomedical imaging modalities, enabling automatic generation of high-quality autofluorescence (AF) reconstructions from hematoxylin and eosin (H&E) stained tissue images.

## Motivation

Digital pathology workflows often require complementary staining methods to reveal different biological structures.
However, certain staining techniques — including autofluorescence — are:

- expensive and time-consuming
- destructive to tissue
- not always available in routine clinical settings

By learning the translation from H&E → AF using paired samples, the model creates virtual AF reconstructions, improving diagnostic access while preserving tissue samples.

## Goal

The aim of this work is to build a robust, high-fidelity stain translation model that:

- preserves microscopic structures and cell morphology
- reproduces texture and fluorescent intensity patterns
- minimizes hallucinated artifacts
- supports future integration into computational pathology pipelines


## Project Highlights

- End-to-end PyTorch Lightning training pipeline

- Multi-objective generator loss (Adversarial + L1 + Perceptual)

- Stable GAN training through manual optimization control

- Reproducible experiment tracking via Weights & Biases

- Real-time visual logging of generated samples during validation

# Model Architecture

This project implements a GAN-based paired image-to-image translation system, inspired by the Pix2Pix framework (Isola et al., 2017).
The goal is to learn a mapping between source images and target ground-truth images using paired supervision — e.g., translating H&E histology images into autofluorescence (AF) stain reconstructions.

The model consists of two adversarial components trained jointly:

Generator: produces realistic target-style images from input images
Discriminator:	distinguishes real target pairs from generator-produced ("fake") pairs

## Generator - U-Net w/ skip connections
The generator follows a U-Net encoder–decoder architecture, which is ideal for pixel-level translation tasks because it preserves spatial information at multiple resolutions.

Design decisions:

Eight encoder and eight decoder blocks: progressively reduce spatial resolution while deepening feature representation, then symmetrically upsample.

Skip connections between encoder and decoder layers preserve high-frequency details (edges, textures) that would otherwise vanish through bottleneck compression.

Tanh activation at the output: maps pixels into $[−1,1]$, improving stability when images are normalized to the same range.

Dropout in early decoder layers: adds stochasticity and mitigates mode collapse during adversarial learning.

Motivation: Traditional ResNet-based generators achieve strong global consistency, but U-Net generators outperform them on translation tasks requiring spatial fidelity (e.g., biomedical imaging, satellite reconstruction).

## Discriminator — Patch-Based Convolutional Critic

The discriminator processes concatenated (input, target) pairs and predicts whether they represent a real correspondence or a generated one.

Design decisions:

Convolutional feature extractor with stride-2 downsampling progressively compresses spatial structure.

LeakyReLU activations avoid dying neuron effects common in GAN training.

No sigmoid at the final layer — logits are used directly with BCEWithLogitsLoss for better numerical stability.

Adaptive average pooling (FC layer instead of a fully convolutional patch classifier): this design allows the discriminator to make a global decision while still attending to patch-level correspondence.

Motivation: Patch-wise supervision encourages high-frequency realism, while global pooling enforces correct large-scale structure and pigment distribution.

## Loss Functions

The generator is trained using a multi-objective loss ensuring realism, perceptual quality, and pixel-level similarity:
	
- Adversarial Loss (BCEWithLogitsLoss):	Fool the discriminator, drives realism and structural correctness
- L1 Reconstruction Loss: 	Pixel-level similarity, enforces accurate spatial alignment & intensity mapping
- Perceptual Loss (VGG-16 relu1_2):	Feature-space similarity, encourages texture consistency & avoids over-smoothed translations

Weighting is tuned to balance stability vs. fidelity:
$$
G_loss = g_adv + \lambda_L1 * L1 + \lambda_perceptual * perceptual
$$

Adding perceptual loss is particularly useful in biomedical imaging: while L1 guarantees correctness, perceptual loss prevents blurry staining and preserves fine cellular texture.

## Optimization and Training Strategy

Adam optimizers with β = (0.5, 0.999), the standard configuration for stable GAN training.

Custom training loop (automatic_optimization=False) to alternate generator and discriminator updates manually, preventing gradient imbalance.

Weight initialization (normal, μ=0, σ=0.02) following DCGAN best practices.

## Logging and Visualization

During validation, generated samples are logged via Weights & Biases.

This enables qualitative inspection of:

- staining consistency

- texture sharpness

- color distribution

- convergence stability over epochs


Simpler dataset:

    https://drive.google.com/drive/folders/1jApbId20lX8AY0tIsoX2_2BHBLPoxD4L
    https://github.com/bupt-ai-cz/BCI

labsyspharm/ORION-CRC dataset HE 18-channels image pairs
https://zenodo.org/records/7637988