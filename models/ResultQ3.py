import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import ( confusion_matrix, accuracy_score, precision_score, recall_score, f1_score )

# import the 3 models
from ANNTrainer import ANNTrainer
from LogisticRegressionTrainer import LogisticRegressionTrainer
from SVMTrainer import SVMTrainer

# Paths of dataset
train_file = '../dataset/train.csv'
test_file  = '../dataset/test.csv'

# Feature lists
cat_columns = ['Category','EntityType','EvidenceRole','SuspicionLevel','LastVerdict',
               'ResourceType','Roles','AntispamDirection','ThreatFamily']
num_columns = ['DeviceId','Sha256','IpAddress','Url','AccountSid','AccountUpn','AccountObjectId',
               'AccountName','DeviceName','NetworkMessageId','EmailClusterId','RegistryKey',
               'RegistryValueName','RegistryValueData','ApplicationId','ApplicationName',
               'OAuthApplicationId','FileName','FolderPath','ResourceIdName','OSFamily',
               'OSVersion','CountryCode','State','City']

# Training
def run_trainer(TrainerClass, use_time):
    model = TrainerClass(
        train_df=pd.read_csv(train_file, low_memory=False),
        test_df=pd.read_csv(test_file, low_memory=False),
        categorical_features=cat_columns.copy(),
        numerical_features=num_columns.copy(),
        time_feature=use_time
    )
    model.prepare_data()

    # ANNTrainer：find_best_alpha()
    if hasattr(model, "find_best_alpha") and callable(getattr(model, "find_best_alpha")):
        model.find_best_alpha()

    # SVM / Logistic Regression：find_best_c()
    if hasattr(model, "find_best_c") and callable(getattr(model, "find_best_c")):
        model.find_best_c()

    model.train()
    model.predict()
    return model

# Compute metrics
def compute_metrics(y_true, y_pred):
    return {
        "acc":  accuracy_score(y_true, y_pred),
        "prec": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "rec":  recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1":   f1_score(y_true, y_pred, average="macro", zero_division=0),
    }

# Print outcome for 3 models
def print_outcome(name, tag, metr):
    print(f"[{name} | {tag}] "
          f"Accuracy: {metr['acc']:.4f} | "
          f"Macro-Precision: {metr['prec']:.4f} | "
          f"Macro-Recall: {metr['rec']:.4f} | "
          f"Macro-F1: {metr['f1']:.4f}")

# Train 3 models with (no_time, with_time), collect predictions and metrics.
def train_all_models():
    results = {}
    for name, Trainer in [("ANN", ANNTrainer),
                          ("LogReg", LogisticRegressionTrainer),
                          ("SVM", SVMTrainer)]:
        # No time
        m0 = run_trainer(Trainer, use_time=False)
        metrics0 = compute_metrics(m0.y_test, m0.y_predict)
        print_outcome(name, "No Time", metrics0)
        # With time
        m1 = run_trainer(Trainer, use_time=True)
        metrics1 = compute_metrics(m1.y_test, m1.y_predict)
        print_outcome(name, "With Time", metrics1)

        results[name] = {
            "no_time":  {"y_true": m0.y_test, "y_pred": m0.y_predict, "metrics": metrics0},
            "with_time":{"y_true": m1.y_test, "y_pred": m1.y_predict, "metrics": metrics1},
        }
    return results


# Draw confusion matrices
def summary_box(ax, metr):
    ax.text(0.02, 0.98,
            f'Acc={metr["acc"]:.3f}\nMacro-F1={metr["f1"]:.3f}',
            transform=ax.transAxes, va='top', ha='left',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8), fontsize=9)

def annotate_confusion(ax, cm, show_norm=False):
    row_sums = cm.sum(axis=1, keepdims=True).astype(float)
    with np.errstate(divide='ignore', invalid='ignore'):
        cm_norm = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums!=0)
    vmax = cm.max() if cm.size > 0 else 1
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            txt = f'{val:d}' if not show_norm else f'{val:d}\n({cm_norm[i, j]*100:.1f}%)'
            color = 'white' if val > (vmax * 0.6) else 'black'
            ax.text(j, i, txt, ha='center', va='center', color=color, fontsize=9)

def compare_confusion_matrices(results, show_norm=False):
    """
    Use the cached predictions in `results` to draw No-Time vs With-Time matrices.
    """
    for name in ["ANN", "LogReg", "SVM"]:
        r0 = results[name]["no_time"]
        r1 = results[name]["with_time"]

        labels = sorted(np.unique(r0["y_true"]))  # assume same label set
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # No Time
        cm0 = confusion_matrix(r0["y_true"], r0["y_pred"], labels=labels)
        im0 = axes[0].imshow(cm0, interpolation='nearest', aspect='auto')
        axes[0].set_title(f'{name} — No Time')
        axes[0].set_xlabel('Predicted'); axes[0].set_ylabel('Actual')
        axes[0].set_xticks(range(len(labels))); axes[0].set_yticks(range(len(labels)))
        axes[0].set_xticklabels(labels, rotation=0); axes[0].set_yticklabels(labels)
        axes[0].figure.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
        annotate_confusion(axes[0], cm0, show_norm=show_norm)
        summary_box(axes[0], r0["metrics"])

        # With Time
        cm1 = confusion_matrix(r1["y_true"], r1["y_pred"], labels=labels)
        im1 = axes[1].imshow(cm1, interpolation='nearest', aspect='auto')
        axes[1].set_title(f'{name} — With Time')
        axes[1].set_xlabel('Predicted'); axes[1].set_ylabel('Actual')
        axes[1].set_xticks(range(len(labels))); axes[1].set_yticks(range(len(labels)))
        axes[1].set_xticklabels(labels, rotation=0); axes[1].set_yticklabels(labels)
        axes[1].figure.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
        annotate_confusion(axes[1], cm1, show_norm=show_norm)
        summary_box(axes[1], r1["metrics"])

        fig.suptitle(f'Confusion Matrices: {name}')
        plt.tight_layout()
        plt.show()

# Draw metric plots
def build_metric_table(results):
    out = {}
    for name in ["SVM", "LogReg", "ANN"]:
        out[name] = {
            "no_time":  results[name]["no_time"]["metrics"],
            "with_time":results[name]["with_time"]["metrics"]
        }
    return out

# Draw P/R/F1 grouped bars, Δ Macro-F1, Accuracy bars
def plot_overall_metric_comparisons(results):
    metrics = build_metric_table(results)
    model_list = ["SVM", "LogReg", "ANN"]
    stat_names = ["prec", "rec", "f1"]
    stat_titles = ["Precision", "Recall", "F1 (Macro)"]

    # P/R/F1 grouped bars for 3 model (with_time - no_time)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for i, model in enumerate(model_list):
        ax = axes[i]
        no_vals  = [metrics[model]["no_time"][k]  for k in stat_names]
        yes_vals = [metrics[model]["with_time"][k] for k in stat_names]
        x = np.arange(len(stat_names))
        width = 0.35
        ax.bar(x - width/2, no_vals,  width=width, label="No time")
        ax.bar(x + width/2, yes_vals, width=width, label="With time")
        ax.set_xticks(x)
        ax.set_xticklabels(stat_titles, rotation=0)
        ax.set_ylim(0, 1.0)
        ax.set_title(f"{model} — P/R/F1")
        if i == 0:
            ax.legend()
    fig.suptitle("Per-Model Comparison of Precision / Recall / Macro-F1")
    plt.tight_layout()
    plt.show()

    # Δ Macro-F1 (with_time - no_time)
    deltas = [metrics[m]["with_time"]["f1"] - metrics[m]["no_time"]["f1"] for m in model_list]
    plt.figure()
    bars = plt.bar(model_list, deltas)
    plt.axhline(0, linewidth=1)
    plt.ylabel("Δ Macro-F1 (With − No)")
    plt.title("Impact of Time Features on Macro-F1")
    for b, d in zip(bars, deltas):
        plt.text(b.get_x() + b.get_width()/2, d + (0.002 if d>=0 else -0.002),
                 f"{d:+.4f}", ha="center",
                 va="bottom" if d>=0 else "top", fontsize=9)
    plt.tight_layout()
    plt.show()

    # Accuracy comparison (with_time - no_time)
    x = np.arange(len(model_list))
    width = 0.35
    acc_no  = [metrics[m]["no_time"]["acc"]  for m in model_list]
    acc_yes = [metrics[m]["with_time"]["acc"] for m in model_list]
    plt.figure()
    plt.bar(x - width/2, acc_no,  width=width, label="No time")
    plt.bar(x + width/2, acc_yes, width=width, label="With time")
    plt.xticks(x, model_list)
    plt.ylim(0, 1.0)
    plt.ylabel("Accuracy")
    plt.title("Overall Accuracy by Model")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Summary computed from results (with_time - no_time)
    def fmt(v): return f"{v:.4f}"
    print("\n Summary (No time → With time)")
    for m in model_list:
        a0,a1 = metrics[m]["no_time"]["acc"],  metrics[m]["with_time"]["acc"]
        p0,p1 = metrics[m]["no_time"]["prec"], metrics[m]["with_time"]["prec"]
        r0,r1 = metrics[m]["no_time"]["rec"],  metrics[m]["with_time"]["rec"]
        f0,f1 = metrics[m]["no_time"]["f1"],   metrics[m]["with_time"]["f1"]
        print(f"{m:>6}: Acc {fmt(a0)}→{fmt(a1)} (Δ{a1-a0:+.4f}) | "
              f"P {fmt(p0)}→{fmt(p1)} (Δ{p1-p0:+.4f}) | "
              f"R {fmt(r0)}→{fmt(r1)} (Δ{r1-r0:+.4f}) | "
              f"F1 {fmt(f0)}→{fmt(f1)} (Δ{f1-f0:+.4f})")

# Main
if __name__ == '__main__':
    # Train the 3 models
    results = train_all_models()

    # Confusion matrices
    compare_confusion_matrices(results, show_norm=False)

    # Metric plots and printed summary
    plot_overall_metric_comparisons(results)
