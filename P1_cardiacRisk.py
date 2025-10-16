#---------------------------------------- support
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import joblib

#---------------------------------------- Data
from  pandas import read_csv
import csv
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

#---------------------------------------- Model
from   sklearn.model_selection import train_test_split
from   sklearn.neural_network  import MLPClassifier
from   sklearn.tree            import DecisionTreeClassifier
from sklearn.pipeline       import Pipeline
from sklearn.preprocessing  import StandardScaler
from sklearn.model_selection import RepeatedStratifiedKFold, GridSearchCV
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.tree           import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes    import GaussianNB
from sklearn.neighbors      import KNeighborsClassifier
from sklearn.linear_model   import LogisticRegression

#---------------------------------------- Evaluation
from sklearn.metrics import (
    confusion_matrix, accuracy_score, recall_score, f1_score,
    roc_auc_score, roc_curve, precision_recall_curve, average_precision_score
)
from sklearn.metrics import make_scorer
import itertools
from scipy.stats import friedmanchisquare, wilcoxon
from statsmodels.stats.multitest import multipletests

#______________________________________________________________
#             DATA
#¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨

#______________________________________________________________
#          Load Data  
#¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨
filename = 'cardiacRisk.csv'

df = read_csv(filename, sep=",")
print(df.describe())

target_col   = df.columns[-1]
feature_cols = df.columns[:-1]

#______________________________________________________________
#          Data  transform - Missing values (-1 trade with median)
#¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨
for col in feature_cols:
    if (df[col] == -1).any():
        med = df.loc[df[col] != -1, col].median()
        df.loc[df[col] == -1, col] = med

if (df[feature_cols] == -1).any().any():
    print("-1 remain after imputation in features.")

# Rebuild arrays AFTER fixing missing values
data = df.values
M    = data.shape[1] - 1
X    = data[:, 0:M].astype(float)
T    = data[:, M].astype(int)
N    = T.size

print(f"Samples: {N} | Features: {M}")
print(f"positives: {T.sum()} ({T.mean():.1%}), negatives: {N - T.sum()} ({1 - T.mean():.1%})")

#______________________________________________________________
#          DATA VISUALIZATION (imbalance + potential outliers)
#¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨
try:
    neg = int((T == 0).sum()); pos = int((T == 1).sum())
    plt.figure(); plt.bar(['No Event','Event'], [neg, pos]); plt.title('Class Balance')
    plt.tight_layout(); plt.savefig('class_balance.png', dpi=150); plt.close()
except Exception as e:
    print(f"Could not plot class balance: {e}")

try:
    for col in feature_cols:
        plt.figure(); plt.boxplot(df[col].astype(float), vert=False, whis=1.5)
        plt.title(f'Boxplot: {col}'); plt.tight_layout()
        plt.savefig(f'boxplot_{col}.png', dpi=130); plt.close()
except Exception as e:
    print(f"Could not create boxplots: {e}")

#______________________________________________________________
#                  DATA SPLIT (70/30 holdout)
#¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨

Xtrain, Xtest, Ttrain, Ttest = train_test_split(
    X, T, test_size=0.30, stratify=T, random_state=42
)

print(f"Train: {len(Ttrain)} ({Ttrain.mean():.1%} +)")
print(f"Test : {len(Ttest)} ({Ttest.mean():.1%} +)")

#______________________________________________________________
#      TRAINING & CROSS-VALIDATION
#  - Tune on TRAIN via Stratified CV (no TEST usage)
#  - SMOTE lives INSIDE the pipeline to avoid leakage
#  - Scaling inside pipeline for gradient/distance-based models
#  - Store tuned models for later TEST evaluation
#¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨

RANDOM_STATE = 42
USE_SMOTE = True 

def build_pipe(estimator, needs_scale):
    steps = []
    if USE_SMOTE:
        steps.append(('smote', SMOTE(random_state=RANDOM_STATE)))
    if needs_scale:
        steps.append(('scaler', StandardScaler()))
    steps.append(('clf', estimator))
    if USE_SMOTE:
        return ImbPipeline(steps)
    else:
        return Pipeline(steps)

# Models & Param Grids
models_and_grids = {
    "Decision Tree": {
        "pipe": build_pipe(
            DecisionTreeClassifier(
                random_state=RANDOM_STATE,
                class_weight=None  # class balance is handled by SMOTE when USE_SMOTE=True
            ),
            needs_scale=False
        ),
        "param_grid": {
            "clf__criterion": ["gini", "entropy"],
            "clf__max_depth": [3, 5, 7, 9, None],
            "clf__min_samples_leaf": [1, 3, 5, 10, 20],
            "clf__ccp_alpha": np.linspace(0.0, 0.02, 5)
        }
    },
    "Neural Network": {
        "pipe": build_pipe(
            MLPClassifier(
                max_iter=3000, solver="adam", activation="relu",
                early_stopping=True, validation_fraction=0.15,
                n_iter_no_change=20, random_state=RANDOM_STATE
            ),
            needs_scale=True
        ),
        "param_grid": {
            "clf__hidden_layer_sizes": [(16,), (32,), (64,), (32,16)],
            "clf__alpha": [1e-5, 1e-4, 1e-3, 1e-2],
            "clf__learning_rate_init": [1e-3, 5e-4, 1e-4],
            "clf__batch_size": [32, 64, 128]
        }
    },
    "Logistic Regression": {
        "pipe": build_pipe(
            LogisticRegression(
                max_iter=5000, solver="liblinear" 
            ),
            needs_scale=True
        ),
        "param_grid": {
            "clf__C": [0.01, 0.03, 0.1, 0.3, 1, 3, 10],
            "clf__penalty": ["l1", "l2"]
        }
    },
    "KNN": {
        "pipe": build_pipe(
            KNeighborsClassifier(),
            needs_scale=True
        ),
        "param_grid": {
            "clf__n_neighbors": [3, 5, 7, 9, 11, 13, 15, 17, 21],
            "clf__weights": ["uniform", "distance"],
            "clf__p": [1, 2]  # Manhattan vs Euclidean
        }
    },
    "Bayesian": {
        "pipe": build_pipe(
            GaussianNB(),
            needs_scale=True
        ),
        "param_grid": None
    }
}

# Cross-validation on TRAIN only
inner_cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=RANDOM_STATE)
scoring = {
    "roc_auc": "roc_auc",
    "sensitivity": make_scorer(recall_score, pos_label=1),
    "specificity": make_scorer(recall_score, pos_label=0),
    "f1": "f1",
    "accuracy": "accuracy",
}
REFIT_METRIC = "sensitivity" 
trained_models = {}  
cv_report     = {}  
gridsearch_objs = {} 

print("\nCROSS-VALIDATION on TRAIN")
for name, spec in models_and_grids.items():
    pipe, grid = spec["pipe"], spec["param_grid"]

    if grid is None:
        # Baseline GaussianNB: just cross_val_score equivalent via GridSearchCV workaround
        gs = GridSearchCV(
            estimator=pipe,
            param_grid={"clf__var_smoothing": [1e-9]},  # dummy single point
            scoring=scoring,
            refit=REFIT_METRIC,
            cv=inner_cv,
            n_jobs=-1,
            verbose=0
        )
    else:
        gs = GridSearchCV(
            estimator=pipe,
            param_grid=grid,
            scoring=scoring,
            refit=REFIT_METRIC,
            cv=inner_cv,
            n_jobs=-1,
            verbose=0
        )

    gs.fit(Xtrain, Ttrain)
    trained_models[name] = gs.best_estimator_
    gridsearch_objs[name] = gs

    best_idx = gs.best_index_  # index of the best param set according to REFIT_METRIC
    cv_means = {
        "cv_mean_roc_auc":     gs.cv_results_["mean_test_roc_auc"][best_idx],
        "cv_mean_sensitivity": gs.cv_results_["mean_test_sensitivity"][best_idx],
        "cv_mean_specificity": gs.cv_results_["mean_test_specificity"][best_idx],
        "cv_mean_f1":          gs.cv_results_["mean_test_f1"][best_idx],
        "cv_mean_accuracy":    gs.cv_results_["mean_test_accuracy"][best_idx],
    }

    cv_report[name] = {
        "best_params": gs.best_params_,
        "refit_metric": REFIT_METRIC,
        **cv_means
    }

    print(
        f"{name:20s} "
        f"[refit={REFIT_METRIC}] "
        f"CV AUC={cv_means['cv_mean_roc_auc']:.3f} | "
        f"CV SE={cv_means['cv_mean_sensitivity']:.3f} | "
        f"CV SP={cv_means['cv_mean_specificity']:.3f} | "
        f"CV F1={cv_means['cv_mean_f1']:.3f} | "
        f"CV ACC={cv_means['cv_mean_accuracy']:.3f} | "
        f"best_params={gs.best_params_}"
    )

print("\nModels tuned via CV on TRAIN. Store `trained_models` for the TEST Evaluation block.")

# Exports
os.makedirs("models", exist_ok=True)
os.makedirs("cv_outputs", exist_ok=True)

# 1) Compact CV summary across models
with open("cv_outputs/cv_summary_table.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["Model","refit_metric","CV_AUC","CV_SE","CV_SP","CV_F1","CV_ACC","best_params"])
    for m, info in cv_report.items():
        w.writerow([
            m, info["refit_metric"],
            f"{info['cv_mean_roc_auc']:.4f}",
            f"{info['cv_mean_sensitivity']:.4f}",
            f"{info['cv_mean_specificity']:.4f}",
            f"{info['cv_mean_f1']:.4f}",
            f"{info['cv_mean_accuracy']:.4f}",
            info["best_params"]
        ])
print("Saved: cv_outputs/cv_summary_table.csv")

# 2) Full cv_results_ per model (all hyperparam combos + all metrics)
for m, gs in gridsearch_objs.items():
    df_cv = pd.DataFrame(gs.cv_results_)
    outp = f"cv_outputs/cv_results_{m.replace(' ','_')}.csv"
    df_cv.to_csv(outp, index=False)
    print(f"Saved: {outp}")

# 3) Persist tuned models (binary) for later reuse
for m, est in trained_models.items():
    outp = f"models/{m.replace(' ','_')}.joblib"
    joblib.dump(est, outp)
    print(f"Saved tuned model: {outp}")


#______________________________________________________________
#                     EVALUATION (TEST ONLY)
#¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨

os.makedirs("eval_outputs", exist_ok=True)

test_metrics_rows = []    
test_probs_cache  = {}  

roc_curves = []           
pr_curves  = []         

print("\nTEST EVALUATION")
for name, model in trained_models.items():
    y_prob = model.predict_proba(Xtest)[:, 1]
    y_pred = (y_prob >= 0.50).astype(int)

    tn, fp, fn, tp = confusion_matrix(Ttest, y_pred, labels=[0,1]).ravel()
    se  = tp/(tp+fn) if (tp+fn) else 0.0             
    sp  = tn/(tn+fp) if (tn+fp) else 0.0
    acc = accuracy_score(Ttest, y_pred)
    f1  = f1_score(Ttest, y_pred, pos_label=1)
    auc = roc_auc_score(Ttest, y_prob)
    prec, rec, _ = precision_recall_curve(Ttest, y_prob)

    test_metrics_rows.append([name, auc, se, sp, f1, acc])
    test_probs_cache[name] = (Ttest.copy(), y_prob.copy())

    print(f"{name:20s}  AUC={auc:.3f}  SE={se:.3f}  SP={sp:.3f}  F1={f1:.3f}  ACC={acc:.3f}")

    try:
        cm = np.array([[tn, fp],[fn, tp]])
        fig, ax = plt.subplots()
        im = ax.imshow(cm, cmap='Blues')
        for i in range(2):
            for j in range(2):
                ax.text(j, i, cm[i, j], ha='center', va='center', fontsize=11)
        ax.set_xticks([0,1]); ax.set_yticks([0,1])
        ax.set_xticklabels(['Pred 0','Pred 1']); ax.set_yticklabels(['True 0','True 1'])
        ax.set_title(f'Confusion Matrix (TEST) - {name}')
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        plt.tight_layout()
        plt.savefig(f"eval_outputs/confmat_test_{name.replace(' ','_')}.png", dpi=150)
        plt.close()
    except Exception as e:
        print(f"Could not plot confusion matrix for {name}: {e}")

    fpr, tpr, _ = roc_curve(Ttest, y_prob)
    roc_curves.append((name, fpr, tpr, auc))
    pr_curves.append((name, rec, prec))

# CSV
with open("eval_outputs/test_metrics_table.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["Model","AUC","SE","SP","F1","ACC"])
    for row in sorted(test_metrics_rows, key=lambda r: r[1], reverse=True):
        w.writerow(row)

# Save raw y_true & y_prob per model for statistical tests
for name, (y_true, y_prob) in test_probs_cache.items():
    out = np.column_stack([y_true, y_prob])
    np.savetxt(f"eval_outputs/test_probs_{name.replace(' ','_')}.csv",
               out, delimiter=",", header="y_true,y_prob", comments="", fmt="%.6f")

# Combined ROC plot (TEST)
plt.figure()
for name, fpr, tpr, auc in roc_curves:
    plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
plt.plot([0,1],[0,1],'--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.title('ROC Curves on TEST (all models)')
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig("eval_outputs/roc_test_all.png", dpi=150)
plt.close()

# Combined Precision–Recall plot (TEST)
plt.figure()
for name, rec, prec in pr_curves:
    plt.plot(rec, prec, label=f"{name}")
plt.xlabel('Recall (Sensitivity)')
plt.ylabel('Precision')
plt.title('Precision–Recall on TEST (all models)')
plt.legend(loc='lower left')
plt.tight_layout()
plt.savefig("eval_outputs/pr_test_all.png", dpi=150)
plt.close()

print("\nTest evaluation complete.")

#______________________________________________________________
#        STATISTICAL TESTS (Paired Bootstrap + Friedman + Wilcoxon-Holm)
#        - Works with a single fixed TEST set
#        - Paired bootstrap keeps subjects aligned across models
#        - Metrics compared: AUC, Sensitivity (SE), Specificity (SP)
#¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨

BOOTSTRAP_SEED = 42    
B = 1000                # number of bootstrap replicates
METRICS_DIR = "eval_outputs"

np.random.seed(BOOTSTRAP_SEED)
os.makedirs(METRICS_DIR, exist_ok=True)

print("\nSTATISTICAL TESTS")

# ----------------------- Load test artifacts -----------------------
#  - eval_outputs/test_metrics_table.csv         (summary)
#  - eval_outputs/test_probs_<Model>.csv         (columns: y_true,y_prob)
summary_path = os.path.join(METRICS_DIR, "test_metrics_table.csv")
metrics_tbl  = pd.read_csv(summary_path)

models = metrics_tbl["Model"].tolist()
print(f"Models detected: {models}")

# Load per-model (y_true, y_prob)
prob_data = {}
for name in models:
    f = os.path.join(METRICS_DIR, f"test_probs_{name.replace(' ', '_')}.csv")
    arr = np.loadtxt(f, delimiter=",", skiprows=1)
    y_true, y_prob = arr[:, 0].astype(int), arr[:, 1]
    prob_data[name] = (y_true, y_prob)

# Sanity: all models must share the exact same y_true (paired setting)
y_ref = next(iter(prob_data.values()))[0]
for n, (yt, _) in prob_data.items():
    assert np.array_equal(yt, y_ref), f"y_true mismatch for model: {n}"
y_true_test = y_ref
N = y_true_test.size
print(f"TEST size: {N}")

# Metric helpers
def metric_auc(y, p):
    return roc_auc_score(y, p)

def metric_se_sp(y, p, thr=0.5):
    yhat = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yhat, labels=[0, 1]).ravel()
    se = tp / (tp + fn) if (tp + fn) else 0.0
    sp = tn / (tn + fp) if (tn + fp) else 0.0
    return se, sp

# Paired bootstrap over TEST
def bootstrap_distributions(B=1000):
    auc_mat = np.zeros((B, len(models)))
    se_mat  = np.zeros((B, len(models)))
    sp_mat  = np.zeros((B, len(models)))
    for b in range(B):
        idx = np.random.randint(0, N, size=N)   # resample subjects with replacement
        yb  = y_true_test[idx]
        for j, name in enumerate(models):
            _, p = prob_data[name]
            pb = p[idx]
            auc_mat[b, j] = metric_auc(yb, pb)
            se, sp = metric_se_sp(yb, pb, thr=0.5)  # same threshold used in Evaluation
            se_mat[b, j]  = se
            sp_mat[b, j]  = sp
    return auc_mat, se_mat, sp_mat

print("Running paired bootstrap")
auc_mat, se_mat, sp_mat = bootstrap_distributions(B=B)

np.save(os.path.join(METRICS_DIR, "boot_auc.npy"), auc_mat)
np.save(os.path.join(METRICS_DIR, "boot_se.npy"),  se_mat)
np.save(os.path.join(METRICS_DIR, "boot_sp.npy"),  sp_mat)

# Friedman + Wilcoxon-Holm 
def compare_matrix(mat: np.ndarray, metric_name: str):
    print(f"\n{metric_name}")
    # Global Friedman: needs one array per model, each with B observations
    cols = [mat[:, j] for j in range(mat.shape[1])]
    fr_stat, fr_p = friedmanchisquare(*cols)
    print(f"Friedman chi2={fr_stat:.3f}, p={fr_p:.6f}")

    # Pairwise Wilcoxon (paired) with Holm correction
    pairs, pvals = [], []
    for (i, m1), (j, m2) in itertools.combinations(enumerate(models), 2):
        stat, p = wilcoxon(mat[:, i], mat[:, j], zero_method='wilcox', alternative='two-sided')
        pairs.append((m1, m2)); pvals.append(p)

    reject, p_corr, _, _ = multipletests(pvals, method="holm")
    df_out = pd.DataFrame({
        "Model1": [a for a, b in pairs],
        "Model2": [b for a, b in pairs],
        "p_raw":  pvals,
        "p_holm": p_corr,
        "Significant": reject
    })
    out_csv = os.path.join(METRICS_DIR, f"pairwise_wilcoxon_{metric_name.lower()}.csv")
    df_out.to_csv(out_csv, index=False)
    print(f"Saved pairwise Wilcoxon-Holm: {out_csv}")
    return fr_p, df_out

p_f_auc, auc_w = compare_matrix(auc_mat, "AUC")
p_f_se,  se_w  = compare_matrix(se_mat,  "SE")
p_f_sp,  sp_w  = compare_matrix(sp_mat,  "SP")

# Select and record a "clinically preferred" model
tbl = pd.read_csv("eval_outputs/test_metrics_table.csv")  # has AUC, SE, SP, etc.
# Rule: prioritize SE, break ties by AUC, then by SP
tbl_sorted = tbl.sort_values(by=["SE","AUC","SP"], ascending=[False, False, False])
winner_row = tbl_sorted.iloc[0]
winner_name = winner_row["Model"]

tbl_sorted.to_csv("eval_outputs/test_metrics_ranked_by_SE_AUC_SP.csv", index=False)
with open("eval_outputs/selected_model.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["Selected_Model","Reason"])
    w.writerow([winner_name, "Rule: maximize SE, then AUC, then SP"])

print(f"Saved ranked table and selected model: {winner_name}")

print("\nStatistical comparison complete.")
