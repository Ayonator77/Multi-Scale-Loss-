# Multi-Scale Loss Function for Deep Neural Network Classification

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Ayonator77/Multi-Scale-Loss-/blob/main/MultiScaleLossGPU.ipynb)

A GPU-accelerated PyTorch implementation of the **two-scale loss function** introduced in:

> Berlyand, L., Creese, R., & Jabin, P.-E. (2024). *A novel multi-scale loss function for classification problems in machine learning.* **Journal of Computational Physics**, 498, 112679. [https://doi.org/10.1016/j.jcp.2023.112679](https://doi.org/10.1016/j.jcp.2023.112679) · [arXiv:2106.02676](https://arxiv.org/abs/2106.02676)

---

## Overview

Standard cross-entropy loss treats every misclassified example the same. This implementation explores a fundamentally different idea: **focus gradient updates on the samples that are hardest to classify** by applying a larger effective scale to well-classified examples and a smaller scale to poorly-classified ones. The result is a loss function that automatically directs training effort where it matters most.

The two-scale loss splits each mini-batch into two groups based on a **classification confidence threshold η**:

- Samples where the correct class logit exceeds the second-best logit by less than η → scaled by **R₁** (smaller)
- Samples where it exceeds by η or more → scaled by **R₂** (larger, R₂ >> R₁)

Both R₁ and R₂ are **learnable parameters**, allowing the scales to adapt jointly with the network weights via SGD. This is the key advantage over fixed-scale variants: the scales self-correct throughout training.

---

## Key Results (from the paper)

| Dataset   | Network     | Accuracy gain (train) | Accuracy gain (test) | Top-k / Close-enough gain |
|-----------|-------------|----------------------|----------------------|---------------------------|
| MNIST     | FC (2-layer)| +7.1%                | +6.7%                | —                         |
| CIFAR-10  | LeNet-5     | +12%                 | +1.5%                | Consistently positive     |
| CIFAR-100 | LeNet-5     | +13.4%               | ~0.1%                | **+10% close-enough** @ aₜ=0.1 |
| CIFAR-100 | WRN + ASAM  | —                    | -0.5% (overall acc)  | **+10% close-enough** @ aₜ=0.1 |

The most striking result is on harder datasets like CIFAR-100: while raw accuracy changes little on the test set, the **distribution of misclassified examples shifts dramatically**. The two-scale network nearly eliminates examples that are egregiously wrong, concentrating misclassifications close to the decision boundary.

---

## Architecture

Three fully-connected network variants are implemented, all sharing the same two-scale forward-pass logic:

| Class | Layers | Hidden dims | Use case |
|-------|--------|-------------|----------|
| `Net`  | 3      | 784 → 1000 → 500 → 10 | FashionMNIST (shallow) |
| `Net2` | 3      | 784 → 2000 → 3000 → 10 | Wide shallow network |
| `Net3` | 8      | 784 → (1000 ×6) → 10 | Deep residual-style test |

Each network embeds R₁ and R₂ directly as `nn.Parameter` tensors, initialized from the spectral norm product of the weight matrices:

```
R = ‖W₁‖ · ‖W₂‖ · ... · ‖Wₗ‖
R1_init = R,  R2_init = 10 · R
```

The forward pass computes logits, then applies the confidence gate:

```python
# Confidence of correct class over runner-up
r_int = correct_logit - max_other_logit

# Mask: 1 if sample NOT well-classified (δX < η)
q = 1 - threshold_indicator

# Second pass uses this mask as the g argument, selecting R1 or R2
output, _, _ = model(data, target, eta, pass=0, g=q, R=R)
loss = F.nll_loss(output, target)
```

---

## Datasets Supported

| Function           | Dataset      | Epochs | Notes                        |
|--------------------|--------------|--------|------------------------------|
| `main()`           | MNIST        | 50     | Net3, deep FC                |
| `main_kmnist()`    | KMNIST       | 50     | Kuzushiji-MNIST characters   |
| `main_fmnist()`    | FashionMNIST | 200    | Net (shallow), harder task   |
| `main_qmnist()`    | QMNIST       | 50     | Extended MNIST test set      |

All datasets are downloaded automatically via `torchvision.datasets`.

---

## Requirements

```
torch >= 1.11
torchvision
scipy
matplotlib
numpy
easydict
```

Install via:
```bash
pip install torch torchvision scipy matplotlib numpy easydict
```

This notebook is designed to run on a **CUDA-enabled GPU**. Colab's free T4 GPU tier is sufficient for MNIST/KMNIST/FMNIST experiments.

---

## Usage

### Quick Start (Google Colab)

Click the badge at the top to open directly in Colab with GPU runtime enabled.

### Running Locally

```python
# Single-scale baseline (η → ∞, effectively one scale)
p, q, r = main_fmnist(eta=10**4, k=0)

# Two-scale loss (η = 0.01, strong focus on hard samples)
p, q, r = main_fmnist(eta=10**(-2), k=0)
```

**`eta`** (`η`) controls the confidence threshold:
- Small η (e.g. `0.01`) → most samples qualify as "well-classified" early → aggressive two-scale behavior
- Large η (e.g. `10000`) → almost no samples qualify → degenerates to standard cross-entropy
- `eta = 10**(-2)` is the recommended starting point per the paper

**`k`** is the random seed for reproducibility.

### Multi-Seed Evaluation

```python
x, y, y2 = [], [], []
for seed in range(10):
    # Single-scale baseline
    p, q, r = main_fmnist(10**4, seed)
    y.append(q)

    # Two-scale
    p_, q_, r_ = main_fmnist(10**(-2), seed)
    y2.append(q_)

plot_acc(x, y_graph, y2_graph, "Accuracy: Single vs Two-Scale")
```

---

## Output & Checkpointing

After each run, the following files are saved to disk:

```
2seed_{k}_R2_10_epochs_{N}_eta{η}_mom{momentum}acc_2s.pt      # (p, q, r) tensors
2model_seed_{k}_R2_10_epochs_{N}_eta{η}_mom{momentum}acc_2s.pt # model state dict
```

- `p` — per-batch training accuracy list
- `q` — test accuracy sampled every N epochs (11-element tensor)
- `r` — per-epoch classification confidence matrix (shape: `[epochs, batch_size]`)

Load a saved model:

```python
model = Net3()
model.load_state_dict(torch.load('2model_seed_0_R2_10_epochs_50_eta0.01_mom0acc_2s.pt'))
```

---

## Hyperparameter Reference

| Parameter    | Default | Description |
|--------------|---------|-------------|
| `lr`         | 0.1     | SGD learning rate |
| `momentum`   | 0       | SGD momentum (set to 0 in all reported experiments) |
| `batch_size` | 128     | Mini-batch size |
| `epochs`     | 50–200  | Training duration (dataset-dependent) |
| `eta`        | varied  | Classification confidence threshold η |
| `R1_init`    | R       | Initial scale for poorly-classified samples |
| `R2_init`    | 10·R    | Initial scale for well-classified samples |

---

## Background: Why Two Scales?

The core insight is that the softmax cross-entropy loss

```
L(α, s) = -log( exp(R · x̂ᵢ₍ₛ₎) / Σⱼ exp(R · x̂ⱼ) )
```

depends critically on the **norm R of the weight matrices**. For correctly-classified samples, a larger R drives the loss toward zero regardless of the direction of α — meaning gradient updates on well-classified samples mostly grow R rather than improve the decision boundary geometry.

The two-scale loss addresses this by assigning a large R₂ to well-classified samples (freezing their effective learning contribution) and a small R₁ to hard samples (keeping their gradients large and directional). This focuses the network's representational capacity on the tail of the error distribution.

See the paper for the full theoretical treatment, including convergence conjectures and connections to the stability analysis in [Berlyand & Jabin, 2018].

---

## Citation

```bibtex
@article{berlyand2024multiscale,
  title   = {A novel multi-scale loss function for classification problems in machine learning},
  author  = {Berlyand, Leonid and Creese, Robert and Jabin, Pierre-Emmanuel},
  journal = {Journal of Computational Physics},
  volume  = {498},
  pages   = {112679},
  year    = {2024},
  doi     = {10.1016/j.jcp.2023.112679}
}
```

---

## License

This implementation is released for research and educational use. The underlying method is described in the paper above (© 2023 Elsevier Inc.). Please cite the original work if you use this code in your research.
