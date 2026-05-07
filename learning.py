import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


df_train = pd.read_csv("data/train.csv")
df_test = pd.read_csv("data/test.csv")

# Priprema podataka
feature_cols = [c for c in df_train.columns if c not in ["Activity", "ActivityName", "subject"]]

X_train = df_train[feature_cols].values
y_train = df_train["ActivityName"].values

X_test = df_test[feature_cols].values
y_test = df_test["ActivityName"].values


# Skaliranje - važno za LR!
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)   # transform, ne fit_transform!

# Model
lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train_sc, y_train)

# Evaluacija
y_pred = lr.predict(X_test_sc)
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy: {acc:.4f}")
print()
print(classification_report(y_test, y_pred))


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

labels = ["LAYING","SITTING","STANDING","WALKING","WALKING_DOWNSTAIRS","WALKING_UPSTAIRS"]

cm = confusion_matrix(y_test, y_pred, labels=labels)

plt.figure(figsize=(9, 7))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=labels, yticklabels=labels)
plt.title("Konfuziona matrica – Logistička regresija (baseline)", fontsize=13, fontweight="bold")
plt.ylabel("Stvarna klasa")
plt.xlabel("Predviđena klasa")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()


from sklearn.svm import SVC

# Model - RBF kernel kao u originalnom radu
svm = SVC(kernel="rbf", C=10, gamma="scale", random_state=42)
svm.fit(X_train_sc, y_train)

# Evaluacija
y_pred_svm = svm.predict(X_test_sc)
acc_svm = accuracy_score(y_test, y_pred_svm)
print(f"Accuracy: {acc_svm:.4f}")
print()
print(classification_report(y_test, y_pred_svm))

# Konfuziona matrica
cm_svm = confusion_matrix(y_test, y_pred_svm, labels=labels)
plt.figure(figsize=(9, 7))
sns.heatmap(cm_svm, annot=True, fmt="d", cmap="Oranges",
            xticklabels=labels, yticklabels=labels)
plt.title("Konfuziona matrica – SVM (RBF kernel)", fontsize=13, fontweight="bold")
plt.ylabel("Stvarna klasa")
plt.xlabel("Predviđena klasa")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# Random Forest - ne treba skaliranje
rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)

y_pred_rf = rf.predict(X_test)
acc_rf = accuracy_score(y_test, y_pred_rf)
print(f"Random Forest Accuracy: {acc_rf:.4f}")
print(classification_report(y_test, y_pred_rf))

# Gradient Boosting
gbt = GradientBoostingClassifier(n_estimators=200, random_state=42)
gbt.fit(X_train, y_train)

y_pred_gbt = gbt.predict(X_test)
acc_gbt = accuracy_score(y_test, y_pred_gbt)
print(f"Gradient Boosting Accuracy: {acc_gbt:.4f}")
print(classification_report(y_test, y_pred_gbt))

# Tabela poređenja
print("\n" + "="*45)
print(f"{'Model':<25} {'Accuracy':>10}")
print("="*45)
print(f"{'Logistička regresija':<25} {acc:.4f}")
print(f"{'SVM (RBF)':<25} {acc_svm:.4f}")
print(f"{'Random Forest':<25} {acc_rf:.4f}")
print(f"{'Gradient Boosting':<25} {acc_gbt:.4f}")
print("="*45)