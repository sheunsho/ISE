"""
Bug Report Classification — Experiment Script
================================================
This script compares multiple classifiers against the Naive Bayes + TF-IDF
baseline for classifying GitHub bug reports as performance-related (1) or not (0).

Classifiers tested:
  1. Naive Bayes (MultinomialNB) — BASELINE (uniform class prior)
  2. Support Vector Machine (LinearSVC) — with balanced class weights
  3. Random Forest — with balanced class weights
  4. Logistic Regression — with balanced class weights
  5. XGBoost — with scale_pos_weight to handle class imbalance

Datasets: 5 DL framework projects (TensorFlow, PyTorch, Keras, MXNet, Caffe)
Setup:    70/30 train/test split, 30 repeats per project per classifier
Metrics:  Precision, Recall, F1 (binary, positive class = 1)
Stats:    Wilcoxon signed-rank test (α = 0.05) comparing each classifier vs baseline

Two experiments are run per invocation:
  - 'weighted': improved classifiers use class weighting (main experiment in the report).
  - 'default':  improved classifiers use no class imbalance handling (for comparison).
Both sets of results are written to results/; 'weighted' uses the unsuffixed filenames
(matching the figure references in the LaTeX report), 'default' uses a '_default' suffix.
"""

# ============================================================
# 1. IMPORTS
# ============================================================
import os
import re
import warnings
import numpy as np
import pandas as pd
import nltk
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend — saves plots without needing a display window
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.base import clone

# --- Classifiers ---
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier      # Gradient boosted trees — strong on imbalanced data

# Statistical testing
from scipy.stats import wilcoxon

# Download NLTK stopwords (skips if already present)
nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords

# Suppress sklearn convergence/future warnings during batch runs
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)


# ============================================================
# 2. TEXT PREPROCESSING
# ============================================================
# Bug reports are messy: HTML tags, emojis, stopwords, mixed case.
# These functions clean the text so the TF-IDF vectorizer gets
# meaningful words only (e.g., "slow", "memory", "crash").

def remove_html(text):
    """Strip HTML tags like <p>, <br>, <div> etc."""
    return re.compile(r'<.*?>').sub('', text)


def remove_emoji(text):
    """Strip emoji characters — they add no value for text classification."""
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"   # emoticons
        u"\U0001F300-\U0001F5FF"   # symbols & pictographs
        u"\U0001F680-\U0001F6FF"   # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"   # flags
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        "]+", flags=re.UNICODE)
    return emoji_pattern.sub('', text)


# Build the stopword set ONCE (using a set for O(1) lookup instead of a list)
STOP_WORDS = set(stopwords.words('english'))


def remove_stopwords(text):
    """Remove common English words ('the', 'is', 'and'...) that don't help classification."""
    return " ".join(word for word in str(text).split() if word not in STOP_WORDS)


def clean_text(text):
    """Lowercase the text, strip non-alphanumeric chars, collapse extra whitespace."""
    text = re.sub(r"[^A-Za-z0-9(),.!?\'`]", " ", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip().lower()


def preprocess(text):
    """Full cleaning pipeline: HTML → emoji → stopwords → normalise."""
    text = remove_html(text)
    text = remove_emoji(text)
    text = remove_stopwords(text)
    text = clean_text(text)
    return text


# ============================================================
# 3. DATA LOADING
# ============================================================
# Each project CSV has columns including: Title, Body, class (0 or 1).
# We merge Title + Body into a single text field, then clean it.
# If Body is NaN (some reports have no body), we just use the Title.

def load_project_data(project_name, data_dir='datasets'):
    """
    Load and preprocess one project's dataset.

    Returns:
        texts  — pandas Series of cleaned text strings
        labels — pandas Series of binary labels (0 = not perf bug, 1 = perf bug)
    """
    path = os.path.join(data_dir, f'{project_name}.csv')
    df = pd.read_csv(path)

    # Merge Title and Body into one text column
    # If Body is NaN, just use the Title on its own
    df['text'] = df.apply(
        lambda row: row['Title'] + '. ' + row['Body'] if pd.notna(row['Body']) else row['Title'],
        axis=1
    )

    # Apply the full text cleaning pipeline
    df['text'] = df['text'].apply(preprocess)

    # The label column is called 'class' in the CSV (0 or 1)
    labels = df['class']

    print(f"  Loaded {project_name}: {len(df)} reports "
          f"({labels.sum()} positive, {len(df) - labels.sum()} negative, "
          f"{labels.mean()*100:.1f}% positive)")

    return df['text'], labels


# ============================================================
# 4. EXPERIMENT CONFIGURATION
# ============================================================

# --- All 5 DL framework projects from the lab spec ---
PROJECTS = ['tensorflow', 'pytorch', 'keras', 'incubator-mxnet', 'caffe']

# --- Classifiers to compare ---
#
# BASELINE: MultinomialNB with a uniform class prior (used in BOTH modes).
#   - MultinomialNB doesn't support class_weight, so we use class_prior=[0.5, 0.5]
#     instead. This is the Bayesian analogue — it corrects the skewed training
#     distribution through the prior rather than through the loss function.
#   - Without this correction the baseline would predict the majority class
#     almost exclusively and score F1 ≈ 0, trivialising every comparison.
#     This is a baseline correction (not an imbalance-handling technique),
#     so the baseline uses it in both the 'default' and 'weighted' experiments.
#
# IMPROVED APPROACHES — two modes:
#   'weighted' mode — SVM, Random Forest, Logistic Regression use
#     class_weight='balanced', which automatically upweights the minority
#     class during training. XGBoost uses scale_pos_weight set to
#     (negative count) / (positive count), calculated per project.
#
#   'default' mode — no class imbalance handling on the improved classifiers.
#     SVM/RF/LR run with class_weight=None and XGBoost runs with its library
#     default scale_pos_weight=1. This isolates how much the class weighting
#     itself contributes to the weighted results.

def build_classifiers(mode, imbalance_ratio):
    """
    Build the classifier dict for the given experiment mode.

    Args:
        mode: 'weighted' or 'default'.
        imbalance_ratio: (neg / pos) ratio for XGBoost's scale_pos_weight
                         in weighted mode. Ignored in default mode.

    Returns:
        Dict mapping classifier name -> fresh scikit-learn / xgboost instance.
        A new dict is built per project so XGBoost's scale_pos_weight reflects
        that project's actual class distribution.
    """
    if mode == 'weighted':
        return {
            'Naive Bayes (Baseline)': MultinomialNB(class_prior=[0.5, 0.5]),
            'SVM':                    LinearSVC(max_iter=10000, dual='auto',
                                                class_weight='balanced'),
            'Random Forest':          RandomForestClassifier(n_estimators=100,
                                                             random_state=42,
                                                             class_weight='balanced'),
            'Logistic Regression':    LogisticRegression(max_iter=1000,
                                                         class_weight='balanced'),
            'XGBoost':                XGBClassifier(
                                          n_estimators=100,
                                          eval_metric='logloss',
                                          random_state=42,
                                          scale_pos_weight=imbalance_ratio,
                                      ),
        }
    elif mode == 'default':
        return {
            'Naive Bayes (Baseline)': MultinomialNB(class_prior=[0.5, 0.5]),
            'SVM':                    LinearSVC(max_iter=10000, dual='auto'),
            'Random Forest':          RandomForestClassifier(n_estimators=100,
                                                             random_state=42),
            'Logistic Regression':    LogisticRegression(max_iter=1000),
            'XGBoost':                XGBClassifier(
                                          n_estimators=100,
                                          eval_metric='logloss',
                                          random_state=42,
                                      ),
        }
    else:
        raise ValueError(f"Unknown mode: {mode!r}. Expected 'default' or 'weighted'.")


# Modes run (in order) by the main entry point.
MODES = ['default', 'weighted']

# --- Experiment parameters ---
NUM_REPEATS = 30     # Lab spec says ~30 repeats (lecturer's code only did 10)
TEST_SIZE   = 0.3    # Lab spec says 70/30 split (lecturer's code used 80/20)
ALPHA       = 0.05   # Significance level for Wilcoxon test

# --- Output paths ---
# In 'weighted' mode (the main experiment, referenced in the LaTeX report),
# outputs use unsuffixed names: raw_results.csv, summary_results.csv,
# wilcoxon_tests.csv, and {project}_f1_boxplot.png.
# In 'default' mode, outputs are suffixed with '_default' so both experiments
# coexist in results/ without overwriting.
RESULTS_DIR = 'results'
FIGURES_DIR = 'results/figures'


def _output_path(filename, mode):
    """Build an output CSV path for a given filename and mode.
    Weighted mode → unsuffixed. Other modes → '_<mode>' suffix before extension.
    """
    if mode != 'weighted':
        stem, ext = os.path.splitext(filename)
        filename = f'{stem}_{mode}{ext}'
    return os.path.join(RESULTS_DIR, filename)


def _figure_path(project, mode):
    """Build the box plot figure path for a given project and mode."""
    suffix = '' if mode == 'weighted' else f'_{mode}'
    return os.path.join(FIGURES_DIR, f'{project}_f1_boxplot{suffix}.png')


# ============================================================
# 5. MAIN EXPERIMENT LOOP
# ============================================================
# For each project:
#   For each of 30 random splits:
#     - Split data 70/30 (same split for ALL classifiers — fair comparison)
#     - Fit TF-IDF on training data only, transform both train and test
#     - Train each classifier, predict on test set
#     - Record precision, recall, F1 for each classifier
#
# KEY DESIGN DECISIONS:
#   - random_state=run ensures all classifiers see the SAME split in each run.
#     This is REQUIRED for the Wilcoxon paired test to be valid.
#   - TF-IDF is fit on training data only (no data leakage from test set).
#   - clone() creates a fresh untrained classifier each run.
#   - build_classifiers(mode, imbalance_ratio) is called per project, so
#     XGBoost's scale_pos_weight in weighted mode matches that project's
#     actual class ratio (and the mode governs whether the other classifiers
#     use class_weight='balanced' at all).

def run_experiments(mode):
    """
    Run the experiment suite for the given mode and return raw results.

    Args:
        mode: 'weighted' (class weighting on improved classifiers) or
              'default' (no class weighting on improved classifiers).

    Returns:
        Nested dict results[classifier][project] = {'precision': [...],
                                                    'recall': [...],
                                                    'f1': [...]}
        with NUM_REPEATS values per list (one per random-state-paired run).
    """
    # Build a reference dict once (with dummy imbalance_ratio) purely to get
    # classifier names for initialising the results structure. The real
    # classifier instances are rebuilt per project inside the loop below
    # so XGBoost's scale_pos_weight matches each project's class ratio.
    clf_names = list(build_classifiers(mode, imbalance_ratio=1.0).keys())
    results = {
        clf_name: {
            proj: {'precision': [], 'recall': [], 'f1': []}
            for proj in PROJECTS
        }
        for clf_name in clf_names
    }

    for project in PROJECTS:
        print(f"\n{'='*60}")
        print(f"Project: {project}  [mode: {mode}]")
        print(f"{'='*60}")

        # Load and preprocess data for this project
        texts, labels = load_project_data(project)

        # --- Calculate class imbalance ratio ---
        # scale_pos_weight = count(negative) / count(positive).
        # Used by XGBoost in weighted mode (ignored in default mode).
        # e.g., Caffe: 253/33 ≈ 7.67 → missing a bug is 7.67x more costly.
        n_pos = labels.sum()
        n_neg = len(labels) - n_pos
        imbalance_ratio = n_neg / n_pos
        print(f"  Class imbalance ratio (neg/pos): {imbalance_ratio:.2f}")

        # Build fresh classifier instances for THIS project and THIS mode.
        # A fresh dict per project means XGBoost's scale_pos_weight is
        # re-set to the current project's ratio in weighted mode.
        classifiers = build_classifiers(mode, imbalance_ratio)

        for run in range(NUM_REPEATS):
            # --- 70/30 train-test split ---
            # random_state=run → same split for every classifier in this run
            X_train_text, X_test_text, y_train, y_test = train_test_split(
                texts, labels,
                test_size=TEST_SIZE,
                random_state=run
            )

            # --- TF-IDF vectorization ---
            # fit on training text ONLY, then transform both sets.
            # This avoids data leakage (the test vocab doesn't influence training).
            vectorizer = TfidfVectorizer()
            X_train = vectorizer.fit_transform(X_train_text)
            X_test  = vectorizer.transform(X_test_text)

            # --- Train and evaluate each classifier on this split ---
            for clf_name, clf_template in classifiers.items():
                # clone() = fresh untrained copy (don't train on top of a previous run)
                clf = clone(clf_template)

                # Train the classifier
                clf.fit(X_train, y_train)

                # Make predictions on the held-out test set
                y_pred = clf.predict(X_test)

                # --- Metrics ---
                # Binary classification: positive class = 1 (performance bug)
                # We use binary (default) not macro — we care about detecting
                # the positive (minority) class specifically.
                p  = precision_score(y_test, y_pred, zero_division=0)
                r  = recall_score(y_test, y_pred, zero_division=0)
                f1 = f1_score(y_test, y_pred, zero_division=0)

                # Store this run's scores (we need all 30 for the Wilcoxon test later)
                results[clf_name][project]['precision'].append(p)
                results[clf_name][project]['recall'].append(r)
                results[clf_name][project]['f1'].append(f1)

            # Progress update every 10 runs
            if (run + 1) % 10 == 0:
                print(f"  Completed run {run + 1}/{NUM_REPEATS}")

    return results


# ============================================================
# 6. RESULTS — SAVE & DISPLAY
# ============================================================

def save_raw_results(results, mode):
    """
    Save EVERY individual run score to CSV.
    This is the reproducibility evidence — the marker can verify any number
    in the report from this file.
    """
    out_path = _output_path('raw_results.csv', mode)
    rows = []
    for clf_name in results:
        for project in PROJECTS:
            for run_idx in range(NUM_REPEATS):
                rows.append({
                    'classifier': clf_name,
                    'project':    project,
                    'run':        run_idx + 1,
                    'precision':  results[clf_name][project]['precision'][run_idx],
                    'recall':     results[clf_name][project]['recall'][run_idx],
                    'f1':         results[clf_name][project]['f1'][run_idx],
                })

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    print(f"\nRaw results saved to: {out_path}")
    return df


def print_summary_table(results, mode):
    """
    Print and save a summary table: mean ± std for each metric,
    per classifier per project. This table goes directly into the report.
    """
    out_path = _output_path('summary_results.csv', mode)

    print(f"\n{'='*80}")
    print(f"SUMMARY RESULTS — mode: {mode}  (mean ± std over {NUM_REPEATS} runs)")
    print(f"{'='*80}")

    summary_rows = []

    for project in PROJECTS:
        print(f"\n--- {project} ---")
        print(f"  {'Classifier':<30} {'Precision':>14} {'Recall':>14} {'F1':>14}")
        print(f"  {'-'*72}")

        for clf_name in results:
            scores = results[clf_name][project]

            p_mean,  p_std  = np.mean(scores['precision']), np.std(scores['precision'])
            r_mean,  r_std  = np.mean(scores['recall']),    np.std(scores['recall'])
            f1_mean, f1_std = np.mean(scores['f1']),        np.std(scores['f1'])

            print(f"  {clf_name:<30} "
                  f"{p_mean:.4f}±{p_std:.4f}  "
                  f"{r_mean:.4f}±{r_std:.4f}  "
                  f"{f1_mean:.4f}±{f1_std:.4f}")

            summary_rows.append({
                'project':        project,
                'classifier':     clf_name,
                'precision_mean': round(p_mean, 4),
                'precision_std':  round(p_std, 4),
                'recall_mean':    round(r_mean, 4),
                'recall_std':     round(r_std, 4),
                'f1_mean':        round(f1_mean, 4),
                'f1_std':         round(f1_std, 4),
            })

    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(out_path, index=False)
    print(f"\nSummary saved to: {out_path}")
    return df_summary


# ============================================================
# 7. STATISTICAL TESTING — WILCOXON SIGNED-RANK TEST
# ============================================================
# For each project, compare each improved classifier against
# the baseline (Naive Bayes) on their 30 paired F1 scores.
#
# WHY WILCOXON?
#   - Non-parametric: doesn't assume F1 scores follow a normal distribution
#   - Paired: each of the 30 runs uses the SAME split for both classifiers
#   - Standard practice in SE research for this kind of comparison

BASELINE_NAME = 'Naive Bayes (Baseline)'


def run_wilcoxon_tests(results, mode):
    """
    Wilcoxon signed-rank test: each improved classifier vs baseline,
    per project, on F1 scores.
    """
    out_path = _output_path('wilcoxon_tests.csv', mode)

    print(f"\n{'='*80}")
    print(f"WILCOXON SIGNED-RANK TESTS — mode: {mode}  (vs {BASELINE_NAME}, α = {ALPHA})")
    print(f"{'='*80}")

    wilcoxon_rows = []

    for project in PROJECTS:
        print(f"\n--- {project} ---")
        baseline_f1 = results[BASELINE_NAME][project]['f1']

        for clf_name in results:
            if clf_name == BASELINE_NAME:
                continue

            clf_f1 = results[clf_name][project]['f1']

            mean_diff = np.mean(clf_f1) - np.mean(baseline_f1)
            direction = "better" if mean_diff > 0 else "worse"

            try:
                stat, p_value = wilcoxon(clf_f1, baseline_f1)
                significant = "YES" if p_value < ALPHA else "no"
            except ValueError:
                stat, p_value = 0, 1.0
                significant = "no (identical)"

            print(f"  {clf_name:<30} p={p_value:.6f}  "
                  f"sig={significant:<5}  Δ={mean_diff:+.4f} ({direction})")

            wilcoxon_rows.append({
                'project':     project,
                'classifier':  clf_name,
                'vs_baseline': BASELINE_NAME,
                'metric':      'f1',
                'mean_diff':   round(mean_diff, 4),
                'direction':   direction,
                'statistic':   round(stat, 4),
                'p_value':     round(p_value, 6),
                'significant': p_value < ALPHA,
            })

    df_wilcoxon = pd.DataFrame(wilcoxon_rows)
    df_wilcoxon.to_csv(out_path, index=False)
    print(f"\nWilcoxon results saved to: {out_path}")
    return df_wilcoxon


# ============================================================
# 8. VISUALISATION — BOX PLOTS
# ============================================================
# Box plots show the DISTRIBUTION of F1 scores across 30 runs.
# They reveal variance, outliers, and spread — much more informative
# than a bar chart showing just the average.

def plot_results(results, mode):
    """Generate F1 box plots comparing classifiers, one figure per project."""

    for project in PROJECTS:
        fig, ax = plt.subplots(figsize=(10, 5))

        # Gather F1 score lists for each classifier
        data_to_plot = []
        tick_labels = []
        for clf_name in results:
            data_to_plot.append(results[clf_name][project]['f1'])
            short_name = clf_name.replace(' (Baseline)', '\n(Baseline)')
            tick_labels.append(short_name)

        # Draw the box plot
        bp = ax.boxplot(data_to_plot, tick_labels=tick_labels, patch_artist=True)

        # Colour the boxes: grey for baseline, distinct colours for each approach
        colors = ['#CCCCCC', '#4C72B0', '#55A868', '#C44E52', '#8172B2']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        title_suffix = '' if mode == 'weighted' else f' ({mode})'
        ax.set_title(f'F1 Score Distribution — {project}{title_suffix}', fontsize=13)
        ax.set_ylabel('F1 Score', fontsize=11)
        ax.set_xlabel('Classifier', fontsize=11)
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        fig_path = _figure_path(project, mode)
        fig.savefig(fig_path, dpi=150)
        plt.close(fig)
        print(f"  Saved: {fig_path}")

    print(f"\nAll figures saved to: {FIGURES_DIR}/")


# ============================================================
# 9. ENTRY POINT — RUN EVERYTHING
# ============================================================

if __name__ == '__main__':
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    print("Starting experiments...")
    print(f"Config: {NUM_REPEATS} repeats, {TEST_SIZE:.0%} test split, "
          f"{len(PROJECTS)} projects, modes: {MODES}")

    for mode in MODES:
        print("\n" + "#" * 80)
        print(f"#  EXPERIMENT MODE: {mode}")
        print("#" * 80)

        results = run_experiments(mode)
        save_raw_results(results, mode)
        print_summary_table(results, mode)
        run_wilcoxon_tests(results, mode)

        print("\nGenerating figures...")
        plot_results(results, mode)

    print("\n" + "=" * 80)
    print("ALL DONE. Check the results/ directory for:")
    print("  Weighted experiment (main results referenced in the report):")
    print(f"    - {_output_path('raw_results.csv',     'weighted')}")
    print(f"    - {_output_path('summary_results.csv', 'weighted')}")
    print(f"    - {_output_path('wilcoxon_tests.csv',  'weighted')}")
    print(f"    - {FIGURES_DIR}/{{project}}_f1_boxplot.png")
    print("  Default experiment (no class weighting, for comparison):")
    print(f"    - {_output_path('raw_results.csv',     'default')}")
    print(f"    - {_output_path('summary_results.csv', 'default')}")
    print(f"    - {_output_path('wilcoxon_tests.csv',  'default')}")
    print(f"    - {FIGURES_DIR}/{{project}}_f1_boxplot_default.png")
    print("=" * 80)
