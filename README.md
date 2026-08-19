# cervicalcancer
# Local vs. Global Feature Extraction in Automated Cytopathology

Comparing a CNN (ResNet-50) against a Vision Transformer (Swin-Tiny) for cervical cell classification, with Grad-CAM interpretability linking each architecture's attention pattern to its misclassification behavior. Co-authored research, published (IEEE).

## Problem

Pap smear screening relies on manually classifying cervical cells into diagnostic categories — a slow, subjective process. This project benchmarks two very different feature-extraction paradigms (CNN's local receptive fields vs. Transformer's global self-attention) on the same classification task, then uses Grad-CAM to explain *why* one architecture outperforms the other on specific, clinically important edge cases.

## Results

| Architecture | Peak Val Accuracy | Convergence Epoch |
|---|---|---|
| ResNet-50 (CNN) | 97.91% | Epoch 12 |
| Swin-Tiny (Transformer) | 98.11% | Epoch 10 |

**Per-class validation performance (precision / recall / F1):**

| Class | ResNet-50 | Swin-Tiny |
|---|---|---|
| Dyskeratotic | 0.98 / 0.99 / 0.98 | 0.98 / **1.00** / 0.99 |
| Koilocytotic | 0.97 / 0.94 / 0.96 | 0.96 / 0.95 / 0.96 |
| Metaplastic | 0.97 / 0.99 / 0.98 | 0.98 / 0.96 / 0.97 |
| Parabasal | 1.00 / 0.99 / 0.99 | 0.99 / 0.99 / 0.99 |
| Superficial-Intermediate | 0.99 / 0.98 / 0.98 | 0.98 / **1.00** / 0.99 |

**Key finding:** Swin-Tiny achieves perfect recall (1.00) on the Superficial-Intermediate class — clinically significant, since a false negative here means a healthy cell gets missed rather than flagged for triage. Grad-CAM shows ResNet-50 fixates on small local artifacts, while Swin-Tiny captures whole-cell context, which explains the gap on this class.

## Dataset

[SIPaKMeD](https://www.kaggle.com/datasets/prahladmehandiratta/cervical-cancer-largest-dataset-sipakmed) — 5,015 cervical cell images across 5 Bethesda system classes (Dyskeratotic, Koilocytotic, Metaplastic, Parabasal, Superficial-Intermediate). 80/20 train/validation split.

## Approach

- Loaded both models pre-trained on ImageNet-1K, replaced the classification head for 5 classes
- Identical training config for a fair comparison: AdamW, lr=1e-4, weight decay=1e-4, cross-entropy loss, 15 epochs, batch size 32
- Standard augmentation: resize to 224×224, random horizontal/vertical flip, random rotation (±15°)
- Evaluated both models with confusion matrices and full classification reports on the held-out validation set
- Applied Grad-CAM to both architectures (with a reshape transform for Swin's windowed attention output) to visualize *where* each model looks when making a prediction, specifically on a case where ResNet-50 misclassified a Superficial-Intermediate cell as Koilocytotic and Swin-Tiny got it right
- Paired the interpretability results with a clinical cost/triage framing — false positives vs. false negatives carry different real-world costs in a screening pipeline

## Tools

`PyTorch` · `torchvision` (ResNet-50, Swin-Tiny) · `pytorch-grad-cam` · `scikit-learn` (metrics) · `seaborn` / `matplotlib` (visualization) · `pandas`

## Training Configuration

| Setting | Value |
|---|---|
| Input size | 224×224 |
| Batch size | 32 |
| Epochs | 15 |
| Optimizer | AdamW |
| Learning rate | 1e-4 |
| Weight decay | 1e-4 |
| Loss function | Cross-Entropy |

## How to Run

1. Download the [SIPaKMeD dataset](https://www.kaggle.com/datasets/prahladmehandiratta/cervical-cancer-largest-dataset-sipakmed) (or run directly on Kaggle, where this notebook was built)
2. Open `cervicalcancer.ipynb` in Jupyter or Kaggle
3. Run cells top to bottom — trains ResNet-50 first, then Swin-Tiny, then generates confusion matrices, classification reports, and Grad-CAM comparison figures
4. Trained weights save as `resnet50_sipakmed_baseline.pth` and `swin_tiny_sipakmed_baseline.pth`

