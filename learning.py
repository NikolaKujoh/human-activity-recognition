import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# ── Load raw UCI HAR data ────────────────────────────────────
def _uniq(names):
    seen, result = set(), []
    for x in names:
        candidate = x
        while candidate in seen:
            candidate += 'n'
        result.append(candidate)
        seen.add(candidate)
    return result

with open('UCI HAR Dataset/features.txt') as f:
    uniq_features = _uniq([line.split()[1] for line in f])

def load_har(split):
    base = f'UCI HAR Dataset/{split}'
    X = pd.read_csv(f'{base}/X_{split}.txt', sep=r'\s+', header=None, names=uniq_features)
    X['subject'] = pd.read_csv(f'{base}/subject_{split}.txt', header=None).squeeze()
    y = pd.read_csv(f'{base}/y_{split}.txt', header=None).squeeze()
    X['Activity']     = y.values
    X['ActivityName'] = y.map({1: 'WALKING', 2: 'WALKING_UPSTAIRS', 3: 'WALKING_DOWNSTAIRS',
                                4: 'SITTING', 5: 'STANDING', 6: 'LAYING'}).values
    X.columns = (X.columns.str.replace(r'[()]', '', regex=True)
                           .str.replace(r'[-]',  '', regex=True)
                           .str.replace(r'[,]',  '', regex=True))
    return X

print("Ucitavanje podataka...")
df_train = load_har('train')
df_test  = load_har('test')

feature_cols = [c for c in df_train.columns if c not in ["Activity", "ActivityName", "subject"]]
X_train = df_train[feature_cols].values
y_train = df_train["ActivityName"].values
X_test  = df_test[feature_cols].values
y_test  = df_test["ActivityName"].values

# ── Skaliranje ───────────────────────────────────────────────
scaler      = StandardScaler()
X_train_sc  = scaler.fit_transform(X_train)
X_test_sc   = scaler.transform(X_test)

labels = sorted(df_train["ActivityName"].unique())

# ── Helper: confusion matrix plot ────────────────────────────
def plot_cm(cm, title, cmap="Blues", fname=None):
    plt.figure(figsize=(9, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap=cmap,
                xticklabels=labels, yticklabels=labels)
    plt.title(title, fontsize=13, fontweight="bold")
    plt.ylabel("Stvarna klasa")
    plt.xlabel("Predvidena klasa")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    if fname:
        plt.savefig(fname, dpi=120, bbox_inches="tight")
    plt.show()

# ── Logisticka regresija ─────────────────────────────────────
print("\nTreniram Logisticku regresiju...")
lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train_sc, y_train)
y_pred_lr  = lr.predict(X_test_sc)
acc_lr     = accuracy_score(y_test, y_pred_lr)
print(f"Accuracy: {acc_lr:.4f}")
print(classification_report(y_test, y_pred_lr))
plot_cm(confusion_matrix(y_test, y_pred_lr, labels=labels),
        "Konfuziona matrica -- Logisticka regresija", "Blues", "cm_lr.png")

# ── SVM (RBF kernel) ─────────────────────────────────────────
print("\nTreniram SVM...")
svm = SVC(kernel="rbf", C=10, gamma="scale", random_state=42)
svm.fit(X_train_sc, y_train)
y_pred_svm = svm.predict(X_test_sc)
acc_svm    = accuracy_score(y_test, y_pred_svm)
print(f"Accuracy: {acc_svm:.4f}")
print(classification_report(y_test, y_pred_svm))
plot_cm(confusion_matrix(y_test, y_pred_svm, labels=labels),
        "Konfuziona matrica -- SVM (RBF kernel)", "Oranges", "cm_svm.png")

# ── Random Forest ────────────────────────────────────────────
print("\nTreniram Random Forest...")
rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
y_pred_rf  = rf.predict(X_test)
acc_rf     = accuracy_score(y_test, y_pred_rf)
print(f"Accuracy: {acc_rf:.4f}")
print(classification_report(y_test, y_pred_rf))
plot_cm(confusion_matrix(y_test, y_pred_rf, labels=labels),
        "Konfuziona matrica -- Random Forest", "Greens", "cm_rf.png")

# ── Gradient Boosting ────────────────────────────────────────
print("\nTreniram Gradient Boosting...")
gbt = GradientBoostingClassifier(n_estimators=200, random_state=42)
gbt.fit(X_train, y_train)
y_pred_gbt = gbt.predict(X_test)
acc_gbt    = accuracy_score(y_test, y_pred_gbt)
print(f"Accuracy: {acc_gbt:.4f}")
print(classification_report(y_test, y_pred_gbt))
plot_cm(confusion_matrix(y_test, y_pred_gbt, labels=labels),
        "Konfuziona matrica -- Gradient Boosting", "Purples", "cm_gbt.png")

# ── Poredjenje modela ────────────────────────────────────────
print("\n" + "=" * 45)
print(f"{'Model':<25} {'Accuracy':>10}")
print("=" * 45)
for name, acc in [("Logisticka regresija", acc_lr),
                  ("SVM (RBF)",            acc_svm),
                  ("Random Forest",        acc_rf),
                  ("Gradient Boosting",    acc_gbt)]:
    print(f"{name:<25} {acc:.4f}")
print("=" * 45)
