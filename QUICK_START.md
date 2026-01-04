# Quick Reference: CHB-MIT Real Labels & Realistic Evaluation

## One-Command Execution (All Steps)

```powershell
cd D:\epimind; `
python ml/training/audit_chbmit_annotations.py --root D:/epimind/ml/data/chbmit --patients chb01 chb02; `
python ml/training/build_chbmit_windows.py --input D:\epimind\ml\data\chbmit --output D:\epimind\ml\data\chbmit_processed_labeled --patients chb01 chb02; `
python ml/training/create_test_dataset.py --input D:\epimind\ml\data\chbmit_processed_labeled --output D:\epimind\ml\data\chbmit_test_labeled --per-patient 1000; `
python ml/training/evaluate_chbmit_quick.py --data D:\epimind\ml\data\chbmit_test_labeled --export D:\epimind\ml\export\models
```

---

## Data Preparation (Steps 1-3)

Same as before - see ML_REAL_LABELS_GUIDE.md

---

## Evaluation Options

### Option A: Quick Evaluation (Balanced Test Set)

```powershell
cd D:\epimind
python ml/training/evaluate_chbmit_quick.py `
  --data D:\epimind\ml\data\chbmit_test_labeled `
  --export D:\epimind\ml\export\models
```

✅ **Use for:** Quick iteration, development, testing pipeline  
✅ **Speed:** ~60 minutes  
❌ **For panels:** Metrics on balanced data (misleading)

---

### Option B: Realistic Evaluation (Natural Imbalanced Distribution)

**This is production-ready and panel-proof.**

#### Fold 1: Train chb01 → Test chb02

```powershell
cd D:\epimind
python ml/training/evaluate_chbmit_realistic.py `
  --train_patient chb01 `
  --test_patient chb02 `
  --epochs 10 `
  --batch_size 32
```

#### Fold 2: Train chb02 → Test chb01

```powershell
cd D:\epimind
python ml/training/evaluate_chbmit_realistic.py `
  --train_patient chb02 `
  --test_patient chb01 `
  --epochs 10 `
  --batch_size 32
```

#### Or both together:

```powershell
cd D:\epimind; `
python ml/training/evaluate_chbmit_realistic.py --train_patient chb01 --test_patient chb02 --epochs 10; `
python ml/training/evaluate_chbmit_realistic.py --train_patient chb02 --test_patient chb01 --epochs 10
```

✅ **Use for:** Panels, publications, clinical deployment  
✅ **Metrics:** PR-AUC, FP/hour, threshold sweep  
✅ **Distribution:** Real imbalanced (0-5% seizures)  
⚠️ **Speed:** ~120 minutes each fold

---

## Expected Realistic Evaluation Output

### Console Output (excerpt):

```
================================================================================
CHB-MIT REALISTIC EVALUATION
================================================================================
Device: cpu
Train patient: chb01
Test patient: chb02

─────────────────────────────────────────────────────────────────────────────
Test set (FULL DISTRIBUTION, NO BALANCING):
  Total windows: 33,696
  Positive windows: 150 (0.45%)
  Negative windows: 33,546
  Hours represented: 4,687.5 hours
─────────────────────────────────────────────────────────────────────────────

Metrics at threshold=0.50:
  Accuracy:     0.9754
  Precision:    0.3521
  Recall:       0.2667
  Specificity:  0.9845
  F1:           0.3030
  ROC-AUC:      0.8913
  PR-AUC:       0.3456
  FP/hour:      4.23

Threshold Sweep Results:
  Thr   Prec     Rec      Spec     F1       FP/hr   
  ─────────────────────────────────────────────────
  0.05  0.0789  0.9333  0.5234   0.1449    87.23
  0.10  0.1234  0.8667  0.7123   0.2143    56.34
  0.15  0.1856  0.7667  0.8456   0.2897    32.12
  ...

Optimal Operating Thresholds:

BEST_F1:
  Threshold:    0.15
  Recall:       0.7667
  FP/hour:      32.12

HIGH_RECALL_0_90:
  Threshold:    0.05
  Recall:       0.9333
  FP/hour:      87.23

HIGH_SPECIFICITY_0_95:
  Threshold:    0.45
  Specificity:  0.9876
  FP/hour:      3.25

✓ JSON report saved: chbmit_eval_report_realistic_chb01_to_chb02.json
✓ Markdown report saved: chbmit_eval_report_realistic_chb01_to_chb02.md
```

### Output Files:

```
ml/export/models/
├── chbmit_eval_report_realistic_chb01_to_chb02.json    ← Structured metrics
├── chbmit_eval_report_realistic_chb01_to_chb02.md      ← Human-readable report
├── chbmit_eval_report_realistic_chb02_to_chb01.json
└── chbmit_eval_report_realistic_chb02_to_chb01.md
```

---

## Key Metrics in Realistic Evaluation

| Metric | What It Means | Why It Matters |
|--------|---------------|----------------|
| **ROC-AUC** | Probability of ranking positive higher than negative | Overall discriminative ability |
| **PR-AUC** ⭐ | Average precision at different recall levels | **Better for imbalanced data** |
| **Accuracy** | (TP+TN)/(All) | Misleading (99% accurate if always "no seizure") |
| **Recall** | TP/(TP+FN) | % of seizures caught (don't miss seizures!) |
| **FP/hour** ⭐ | False alarms per hour | Deployment feasibility (alert fatigue) |
| **F1** | Balance of precision & recall | Good default metric |

---

## Choosing Operating Threshold

After threshold sweep, pick one:

### High Sensitivity (Catch Seizures)
```
Use: high_recall_0_90
Meaning: Will catch ~90% of seizures
Cost: More false alarms (higher FP/hour)
Best for: Clinical monitoring where missing seizures is unacceptable
```

### Balanced (Default)
```
Use: best_f1
Meaning: Best overall performance
Cost: Moderate false alarms
Best for: General deployment when no specific requirement
```

### Low False Alarms (Minimal Alerts)
```
Use: high_specificity_0_95
Meaning: Only ~5% of alerts are false
Cost: May miss some seizures (~5-10%)
Best for: Patient quality of life (alert fatigue is severe problem)
```

---

## View Reports

```powershell
# View JSON (compact)
Get-Content D:\epimind\ml\export\models\chbmit_eval_report_realistic_chb01_to_chb02.json | ConvertFrom-Json | ConvertTo-Json

# View Markdown (human-readable)
Get-Content D:\epimind\ml\export\models\chbmit_eval_report_realistic_chb01_to_chb02.md
```

---

## Verify Data Quality

Before running evaluation, confirm real labels:

```powershell
cd D:\epimind
python -c "
import numpy as np
for pat in ['chb01', 'chb02']:
    d = np.load(f'ml/data/chbmit_processed_labeled/{pat}.npz')
    y = d['y']
    pos = (y > 0.5).sum()
    total = len(y)
    pct = pos/total*100
    print(f'{pat}: {pos:6d} positive, {total-pos:6d} negative ({pct:.2f}% seizure)')
"
```

Expected output:
```
chb01: [>300] positive, [...] negative (0.5-1.0% seizure)
chb02: [>100] positive, [...] negative (0.3-0.5% seizure)
```

---

## Timing

| Step | Time |
|------|------|
| Audit seizures | 2 sec |
| Preprocess (2 patients) | 20-30 min |
| Create test dataset | 2 min |
| Quick eval (quick, balanced) | 60 min |
| Realistic eval per fold | 60-120 min |
| Both folds realistic | 120-240 min |
| **Total (with realistic)** | **3-4 hours** |

---

## Quick Checklist

- [ ] Data preprocessed with real labels (`chbmit_processed_labeled/` exists)
- [ ] Test dataset created (`chbmit_test_labeled/` exists)
- [ ] Quick evaluation passes (JSON report generated) 
- [ ] Realistic evaluation runs both folds
- [ ] JSON + MD reports saved in `ml/export/models/`
- [ ] Threshold sweep shows clear patterns
- [ ] FP/hour < 100 (otherwise threshold needs tuning)

---

Generated: January 4, 2026

See also:
- **ML_REAL_LABELS_GUIDE.md** - Data preparation steps
- **EVALUATION_REALISTIC.md** - Detailed metrics explanation

