# Bug Report Classification

Automated classification of GitHub bug reports as performance-related or not, across five deep learning framework projects (TensorFlow, PyTorch, Keras, MXNet, Caffe). Compares five classifiers — Naive Bayes (baseline), SVM, Random Forest, Logistic Regression, and XGBoost — using TF-IDF features, 30 repeated 70/30 splits, and Wilcoxon signed-rank significance testing.

## Quick Start

```bash
git clone https://github.com/sheunsho/ISE.git
cd ISE
python -m venv venv
source venv/bin/activate          # macOS/Linux
pip install -r requirements.txt
cd lab1
python main.py
```

Results are written to `lab1/results/`. Expected runtime: 2–5 minutes.

## Documentation

- `requirements.pdf` — dependencies and environment setup
- `manual.pdf` — how to use the tool
- `replication.pdf` — step-by-step replication instructions

## Repository Structure

```
.
├── lab1/
│   ├── main.py         # experiment script
│   ├── datasets/       # bug report CSVs for all five projects
│   └── results/        # raw scores, summary table, Wilcoxon tests, figures
├── requirements.txt
├── requirements.pdf
├── manual.pdf
└── replication.pdf
```
