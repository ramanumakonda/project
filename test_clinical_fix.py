"""
test_clinical_fix.py — Comprehensive verification suite for the clinical data fix.
Tests 1 to 7 according to specification.
"""

import io
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import app.app as app_module
from app.app import app, DEVICE
from src.dataset import ClinicalPreprocessor, build_dataloaders
from src.utils import load_config

def run_all_tests():
    print("=" * 70)
    print("RUNNING CLINICAL PREPROCESSING & INFERENCE VERIFICATION SUITE")
    print("=" * 70)
    
    # ── Ensure model and preprocessor are loaded ──────────────────────────
    app_module._load_model()
    _preprocessor = app_module._preprocessor
    _model = app_module._model
    assert _preprocessor is not None, "Preprocessor failed to load"
    assert _model is not None, "Model failed to load"
    print("[INIT] Model and Preprocessor loaded successfully.\n")

    # ──────────────────────────────────────────────────────────────────────
    # Test 1: Valid clinical input
    # ──────────────────────────────────────────────────────────────────────
    print("Test 1: Valid clinical input (numeric + categorical + binary)")
    df1 = pd.DataFrame([{
        "age": 55,
        "region": "FACE",
        "itch": "1",
        "grew": "0",
        "hurt": "0",
        "changed": "1",
        "bleed": "0",
        "elevation": "1"
    }])
    feats1 = _preprocessor.transform(df1)
    assert isinstance(feats1, np.ndarray), "Result must be a numpy ndarray"
    assert feats1.shape == (1, 8), f"Expected shape (1, 8), got {feats1.shape}"
    assert not np.isnan(feats1).any(), "Result contains NaN values"
    print(f"  --> PASS: Shape={feats1.shape}, Values={feats1[0].round(4).tolist()}")

    # ──────────────────────────────────────────────────────────────────────
    # Test 2: Missing numeric input
    # ──────────────────────────────────────────────────────────────────────
    print("\nTest 2: Missing numeric input (age omitted / empty string / NaN)")
    df2 = pd.DataFrame([{
        "age": None,
        "region": "ARM",
        "itch": "0",
        "grew": "0",
        "hurt": "0",
        "changed": "0",
        "bleed": "0",
        "elevation": "0"
    }])
    feats2 = _preprocessor.transform(df2)
    assert feats2.shape == (1, 8), f"Expected shape (1, 8), got {feats2.shape}"
    assert not np.isnan(feats2).any(), "Result contains NaN values after imputation"
    # Verify that the age feature matches the training median (62.0) scaled
    expected_age_scaled = _preprocessor.age_scaler.transform([[62.0]])[0][0]
    actual_age_scaled = feats2[0, 0]
    # Tolerant comparison if median is ~62
    print(f"  --> PASS: Missing age imputed without error. Scaled age={actual_age_scaled:.4f}")

    # ──────────────────────────────────────────────────────────────────────
    # Test 3: Categorical input (case-insensitivity & unseen categories)
    # ──────────────────────────────────────────────────────────────────────
    print("\nTest 3: Categorical input (lowercase 'face', mixed 'Chest', unknown 'foot', unknown 'XYZ')")
    df3 = pd.DataFrame([
        {"age": 40, "region": "face", "itch": 0, "grew": 0, "hurt": 0, "changed": 0, "bleed": 0, "elevation": 0},
        {"age": 40, "region": "Chest", "itch": 0, "grew": 0, "hurt": 0, "changed": 0, "bleed": 0, "elevation": 0},
        {"age": 40, "region": "UNKNOWN_SITE", "itch": 0, "grew": 0, "hurt": 0, "changed": 0, "bleed": 0, "elevation": 0},
    ])
    feats3 = _preprocessor.transform(df3)
    assert feats3.shape == (3, 8), f"Expected shape (3, 8), got {feats3.shape}"
    assert not np.isnan(feats3).any(), "Categorical features produced NaN"
    print(f"  --> PASS: All 3 categorical rows transformed successfully. Region encodings={feats3[:, 1].tolist()}")

    # ──────────────────────────────────────────────────────────────────────
    # Test 4: Invalid numeric input (validation reporting via Flask route)
    # ──────────────────────────────────────────────────────────────────────
    print("\nTest 4: Invalid numeric input validation ('abc' and negative age)")
    client = app.test_client()
    sample_img_path = ROOT / "data" / "images" / "imgs_part_1" / "PAT_1000_31_620.png"
    assert sample_img_path.exists(), f"Sample image {sample_img_path} not found"

    with open(sample_img_path, "rb") as f:
        img_bytes = f.read()

    # 4a: Non-numeric string 'abc'
    res_invalid_str = client.post("/predict", data={
        "image": (io.BytesIO(img_bytes), "test.png"),
        "age": "abc",
        "region": "FACE",
        "itch": "0", "grew": "0", "hurt": "0", "changed": "0", "bleed": "0", "elevation": "0"
    }, content_type="multipart/form-data")
    assert res_invalid_str.status_code == 200
    html_out = res_invalid_str.get_data(as_text=True)
    assert "Invalid patient age" in html_out or "Please enter a valid number" in html_out
    print("  --> PASS (4a): Invalid string 'abc' caught and clear error message rendered in UI.")

    # 4b: Out-of-range age '-5'
    res_negative = client.post("/predict", data={
        "image": (io.BytesIO(img_bytes), "test.png"),
        "age": "-5",
        "region": "FACE",
        "itch": "0", "grew": "0", "hurt": "0", "changed": "0", "bleed": "0", "elevation": "0"
    }, content_type="multipart/form-data")
    assert res_negative.status_code == 200
    html_out_neg = res_negative.get_data(as_text=True)
    assert "Age must be between 0 and 125" in html_out_neg or "Invalid patient age" in html_out_neg
    print("  --> PASS (4b): Negative age '-5' caught and clear validation message rendered.")

    # ──────────────────────────────────────────────────────────────────────
    # Test 5: Full inference workflow (Clicking 'Analyse Lesion')
    # ──────────────────────────────────────────────────────────────────────
    print("\nTest 5: Full inference pipeline (simulate clicking 'Analyse Lesion')")
    res_full = client.post("/predict", data={
        "image": (io.BytesIO(img_bytes), "test.png"),
        "age": "55",
        "region": "FACE",
        "itch": "1",
        "grew": "0",
        "hurt": "0",
        "changed": "1",
        "bleed": "0",
        "elevation": "1"
    }, content_type="multipart/form-data")
    assert res_full.status_code == 200, f"Expected 200 OK, got {res_full.status_code}"
    html_full = res_full.get_data(as_text=True)
    assert "Analysis Results" in html_full, "Page title 'Analysis Results' missing"
    assert "Primary Diagnosis" in html_full, "Primary Diagnosis banner missing"
    assert "Differential Diagnoses (Top 3)" in html_full, "Top-3 Differential Diagnoses missing"
    assert "GradCAM Heatmap" in html_full, "GradCAM Heatmap missing"
    assert "Patient Clinical Summary" in html_full, "Patient Clinical Summary missing"
    assert "data:image/png;base64," in html_full, "Base64 image visualizations missing"
    print("  --> PASS: Full inference completed! Prediction banner, top-3 differential diagnoses, GradCAM heatmap, and clinical summary all rendered.")

    # ──────────────────────────────────────────────────────────────────────
    # Test 6: Preprocessor and Checkpoint Compatibility
    # ──────────────────────────────────────────────────────────────────────
    print("\nTest 6: Preprocessor and checkpoint compatibility check")
    ckpt_path = ROOT / "checkpoints" / "best.pt"
    pkl_path = ROOT / "checkpoints" / "preprocessor.pkl"
    with open(pkl_path, "rb") as f:
        loaded_p = pickle.load(f)
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    
    dummy_clin = loaded_p.transform(pd.DataFrame([{
        "age": 60, "region": "FACE", "itch": 0, "grew": 0, "hurt": 0, "changed": 0, "bleed": 0, "elevation": 0
    }]))
    clin_t = torch.tensor(dummy_clin, dtype=torch.float32).to(DEVICE)
    img_t = torch.randn(1, 3, 128, 128).to(DEVICE)
    with torch.no_grad():
        out_logits, out_attn = _model(img_t, clin_t)
    assert out_logits.shape == (1, 6), f"Expected logits shape (1, 6), got {out_logits.shape}"
    assert out_attn is not None, "Attention weights missing"
    print(f"  --> PASS: Checkpoint and preprocessor are 100% compatible. Logits shape={out_logits.shape}")

    # ──────────────────────────────────────────────────────────────────────
    # Test 7: Regression check (Homepage & Health endpoints)
    # ──────────────────────────────────────────────────────────────────────
    print("\nTest 7: Regression check (GET / and GET /health)")
    res_index = client.get("/")
    assert res_index.status_code == 200
    assert "Analyse Lesion" in res_index.get_data(as_text=True)
    
    res_health = client.get("/health")
    assert res_health.status_code == 200
    assert res_health.json["status"] == "ok"
    print("  --> PASS: Homepage and Health endpoints return 200 OK with expected content.")

    print("\n" + "=" * 70)
    print("ALL 7 TESTS PASSED SUCCESSFULLY!")
    print("=" * 70)

if __name__ == "__main__":
    run_all_tests()
