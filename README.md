# Cross-Modal Attention Fusion with Parameter-Efficient Fine-Tuning for Multimodal Skin Lesion Classification

> Final-Year Academic Project — PAD-UFES-20 Dataset  
> Extending: "Automated Ensemble Multimodal Machine Learning for Healthcare" (IEEE JBHI)

---

## Overview

This project proposes a **novel multimodal deep learning framework** for 6-class skin lesion classification, combining:

| Component | Implementation |
|---|---|
| **Image Encoder** | EfficientNet-B2 (pretrained, frozen) + LoRA adapters |
| **Clinical Encoder** | 2-layer MLP with region embedding |
| **Fusion** | Bidirectional Cross-Modal Attention |
| **Classifier** | 2-layer head → 6 classes |
| **Deployment** | Flask web app with GradCAM + attention visualisation |

The key innovation over the baseline (AutoPrognosis-M) is the **cross-modal attention fusion**: instead of concatenating image and clinical features, each modality dynamically attends to the other using `nn.MultiheadAttention`, learning *per-sample* cross-modal importance weights.

---

## Project Structure

```
├── configs/
│   └── config.yaml          # All hyperparameters in one place
├── src/
│   ├── dataset.py            # PAD-UFES-20 loader + preprocessing
│   ├── lora.py               # LoRA implementation (from scratch)
│   ├── models/
│   │   ├── image_encoder.py  # EfficientNet-B2 + LoRA
│   │   ├── clinical_encoder.py
│   │   ├── fusion.py         # Cross-modal attention fusion
│   │   └── classifier.py    # Full model assembly
│   ├── train.py              # Training loop
│   ├── evaluate.py           # Evaluation + plots
│   └── explain.py            # GradCAM + attention charts
├── app/
│   ├── app.py                # Flask application
│   ├── templates/            # HTML templates
│   └── static/               # CSS + JS
├── notebooks/
│   └── exploration.ipynb
└── requirements.txt
```

---

## Dataset Setup (PAD-UFES-20)

1. Download from Kaggle: [https://www.kaggle.com/datasets/mahdavi1202/skin-lesion](https://www.kaggle.com/datasets/mahdavi1202/skin-lesion)  
   or Zenodo: [https://zenodo.org/record/4555922](https://zenodo.org/record/4555922)

2. Place files:
```
data/
├── images/          ← all .png lesion images
└── metadata.csv     ← clinical metadata table
```

3. Ensure `metadata.csv` has columns:
   `img_id`, `diagnostic`, `age`, `region`, `itch`, `grew`, `hurt`, `changed`, `bleed`, `elevation`

---

## Installation

```bash
# Create virtual environment
python -m venv venv
venv\Scripts\activate       # Windows
# source venv/bin/activate  # Linux/Mac

pip install -r requirements.txt
```

---

## Training

```bash
# Full training (50 epochs, default config)
python src/train.py

# Debug / smoke test (5 epochs)
python src/train.py --debug

# Custom config
```

Training output:
- Checkpoint saved to `checkpoints/best.pt`
- Preprocessor saved to `checkpoints/preprocessor.pkl`
- TensorBoard logs in `runs/` → run `tensorboard --logdir runs/`

---

## Evaluation

```bash
python src/evaluate.py
```

Outputs saved to `outputs/`:
- `classification_report.txt`
- `summary_metrics.csv`
- `confusion_matrix_raw.png`
- `confusion_matrix_normalised.png`
- `roc_curves.png`

---

## Explainability (CLI)

```bash
python src/explain.py \
    --image data/images/PAT_1.png \
    --clinical "65,face,1,0,0,1,1,0"
```

Generates GradCAM overlay, attention weight chart, clinical importance chart in `outputs/`.

---

## Flask Web App

```bash
# From project root
flask --app app/app.py run

# Or
python app/app.py
```

Open [http://localhost:5000](http://localhost:5000)

**Requirements:** trained checkpoint at `checkpoints/best.pt` and `checkpoints/preprocessor.pkl`.

---

## Architecture

```
Image (224×224)          Clinical (8 features)
      |                          |
EfficientNet-B2            MLP Encoder
  (frozen) + LoRA          (region emb + MLP)
  [rank=4, blocks 5-6]          |
      |                          |
 Visual Embedding         Clinical Embedding
    (256-d)                   (256-d)
      |                          |
      +──── CrossModalAttention ──+
               (bidirectional)
                    |
            Fused Embedding (256-d)
                    |
          Classification Head
                    |
        6-Class Logits (ACK/BCC/MEL/NEV/SCC/SEK)
```

### LoRA Parameter Efficiency

| Configuration | Trainable Params |
|---|---|
| Full fine-tune EfficientNet-B2 | ~7.7 M |
| Frozen + head only | ~0.4 M |
| **LoRA (r=4) + head (ours)** | **~0.55 M** |

---

## 6 Target Classes (PAD-UFES-20)

| Code | Full Name | Type |
|---|---|---|
| ACK | Actinic Keratosis | Pre-malignant |
| BCC | Basal Cell Carcinoma | Malignant |
| MEL | Melanoma | Malignant |
| NEV | Melanocytic Nevus | Benign |
| SCC | Squamous Cell Carcinoma | Malignant |
| SEK | Seborrheic Keratosis | Benign |

---

## Configuration

All hyperparameters python src/train.py --config configs/config.yaml --epochs 30
are in `configs/config.yaml`. Key settings:

```yaml
model:
  lora:
    rank: 4
    alpha: 8
    target_blocks: [5, 6]
  fusion:
    num_heads: 4
    embed_dim: 256
training:
  epochs: 50
  batch_size: 32
  learning_rate: 3.0e-4
```

---

## References

1. Pacheco et al. (2020). *PAD-UFES-20: A skin lesion dataset composed of patient data and clinical images collected from smartphones*. Data in Brief.
2. Imrie et al. (2023). *Automated Ensemble Multimodal Machine Learning for Healthcare*. IEEE Journal of Biomedical and Health Informatics.
3. Hu et al. (2022). *LoRA: Low-Rank Adaptation of Large Language Models*. ICLR 2022.
4. Selvaraju et al. (2017). *Grad-CAM: Visual Explanations from Deep Networks*. ICCV 2017.
5. Vaswani et al. (2017). *Attention Is All You Need*. NeurIPS 2017.

---

## Disclaimer

This system is developed for **academic research purposes only**.  
It is **not** a certified medical device and must not be used for clinical diagnosis.
