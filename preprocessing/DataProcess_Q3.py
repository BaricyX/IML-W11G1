import matplotlib.pyplot as plt
import numpy as np

# Outcome metrics for the 3 models
metrics = {
    "SVM": {
        "no_time":  {"acc":0.4800, "prec":0.3190, "rec":0.4079, "f1":0.3580},
        "with_time":{"acc":0.4855, "prec":0.4283, "rec":0.4102, "f1":0.3629}
    },
    "LogReg": {
        "no_time":  {"acc":0.4789, "prec":0.4621, "rec":0.4297, "f1":0.4218},
        "with_time":{"acc":0.4747, "prec":0.4630, "rec":0.4249, "f1":0.4159}
    },
    "ANN": {
        "no_time":  {"acc":0.6785, "prec":0.7031, "rec":0.6103, "f1":0.6117},
        "with_time":{"acc":0.6953, "prec":0.6913, "rec":0.6444, "f1":0.6518}
    }
}

# Comparison of P/R/F1 scores for each model (with_time - no_time)
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
stat_names = ["prec", "rec", "f1"]
stat_titles = ["Precision", "Recall", "F1 (Macro)"]
model_list = ["SVM", "LogReg", "ANN"]

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

# Increment of Macro-F1 (with_time - no_time)
deltas = [metrics[m]["with_time"]["f1"] - metrics[m]["no_time"]["f1"] for m in model_list]
plt.figure()
bars = plt.bar(model_list, deltas)
plt.axhline(0, linewidth=1)
plt.ylabel("Δ Macro-F1 (With − No)")
plt.title("Impact of Time Features on Macro-F1")
# 在柱子上标注数值
for b, d in zip(bars, deltas):
    plt.text(b.get_x() + b.get_width()/2, d + (0.002 if d>=0 else -0.002),
             f"{d:+.4f}", ha="center", va="bottom" if d>=0 else "top", fontsize=9)
plt.tight_layout()
plt.show()

# Accuracy Comparison (with_time - no_time)
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

# Comparison Summary
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
