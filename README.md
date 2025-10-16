# IML-W11G1

COMP90049 Introduction to Machine Learning - Assignment2

## Link

Dataset Link: [Microsoft Security Incident Prediction](https://www.kaggle.com/datasets/Microsoft/microsoft-security-incident-prediction/data)

GitHub: https://github.com/BaricyX/IML-W11G1

## Directory

```
├── README.md
│
├── dataset/                          
│   ├── sample_100000.csv
│   ├── train.csv
│   └── test.csv
│
├── preprocessing/
│   ├── dataset.py
│   ├── timestamp.py
│   ├── DataProcessStrategy.py
│   ├── mitre_feature_engineering.py
│   └── Data Reproduction Guide.md
│
├── models/
│   ├── classifier_trainer.py
│   ├── LogisticRegressionTrainer.py
│   ├── SVMTrainer.py
│   ├── ANNTrainer.py
│   ├── KMeans.py
│   └── ResultQ3.py
│
├── output/
│   ├── TechniquePatternPlotting.py
│   └──
│
└── Requirements.txt
```



# Quick Start

## 1) Install

```bash
pip install -r Requirements.txt
```

## 2) Commands

- `preprocessing/dataset.py` — Stratified sample and split to `dataset/`.

```bash
python preprocessing/dataset.py --input-csv GUIDE_Test.csv \
  --out-sample dataset/sample_100000.csv \
  --out-train dataset/train.csv --out-test dataset/test.csv \
  --sample-size 100000 --train-size 80000 --test-size 20000 --seed 2025
```

- `preprocessing/DataProcessStrategy.py` — Strategy helpers for (with/without) time features.
   It’s imported by the trainers.
- `preprocessing/mitre_feature_engineering.py` — Build MITRE risk features (expects `all_techniques.csv` in the same folder).

```bash
python mitre_feature_engineering.py    
```

- `preprocessing/Data Reproduction Guide.md` — Steps and command to reproduce stratified sample and train/test split.
  
- `models/classifier_trainer.py` — Base classes and utilities for all trainers.
   You don’t run this file; it’s imported by the trainers.
- `models/KMeans.py` — Cluster MITRE technique patterns on the train set.

```bash
python models/KMeans.py                
```

- `models/LogisticRegressionTrainer.py` — LR trainer (used by `ResultQ3.py`).
   Run via the experiment driver below.
- `models/SVMTrainer.py` — Linear SVM trainer (used by `ResultQ3.py`).
   Run via the experiment driver below.
- `models/ANNTrainer.py` — MLP/ANN trainer (used by `ResultQ3.py`).
   Run via the experiment driver below.
- `models/ResultQ3.py` — One-click experiment: ANN / LR / SVM, with & without time features (reads `dataset/train.csv`, `dataset/test.csv`).

```bash
python models/ResultQ3.py             
```

- `output/TechniquePatternPlotting.py` — Plot MITRE technique risk patterns.

```bash
python -m output.TechniquePatternPlotting
```
