import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

print("=== LOADING DATASET ===")
df = pd.read_csv('cardio_train.csv', sep=';')
print(f"Shape: {df.shape}")
print(df.head())

print("\n=== PREPROCESSING ===")
df['age_years'] = (df['age'] / 365).round()
df = df[(df['ap_hi'] > 0) & (df['ap_lo'] > 0)]
df = df[(df['ap_hi'] < 300) & (df['ap_lo'] < 200)]
df = df[(df['ap_hi'] >= df['ap_lo'])]
df['bmi'] = df['weight'] / ((df['height'] / 100) ** 2)
df = df.drop(columns=['id', 'age'])
print(f"Shape after cleaning: {df.shape}")
print(f"Class distribution:\n{df['cardio'].value_counts()}")

print("\n=== SPLITTING DATA ===")
X = df.drop(columns=['cardio'])
y = df['cardio']
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp)
print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_val_sc = scaler.transform(X_val)

print("\n=== TRAINING MODELS ===")

print("Training Logistic Regression...")
lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train_sc, y_train)
lr_pred = lr.predict(X_val_sc)
lr_prob = lr.predict_proba(X_val_sc)[:,1]

print("Training Random Forest...")
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_val)
rf_prob = rf.predict_proba(X_val)[:,1]

print("Training SVM...")
svm = SVC(kernel='rbf', probability=True, random_state=42)
svm.fit(X_train_sc, y_train)
svm_pred = svm.predict(X_val_sc)
svm_prob = svm.predict_proba(X_val_sc)[:,1]

print("Training XGBoost...")
xgb = XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss')
xgb.fit(X_train, y_train)
xgb_pred = xgb.predict(X_val)
xgb_prob = xgb.predict_proba(X_val)[:,1]

print("\n=== RESULTS ===")
models = {
    'Logistic Regression': (lr_pred, lr_prob),
    'Random Forest':       (rf_pred, rf_prob),
    'SVM':                 (svm_pred, svm_prob),
    'XGBoost':             (xgb_pred, xgb_prob),
}

results = []
for name, (pred, prob) in models.items():
    acc  = accuracy_score(y_val, pred)
    prec = precision_score(y_val, pred)
    rec  = recall_score(y_val, pred)
    f1   = f1_score(y_val, pred)
    auc  = roc_auc_score(y_val, prob)
    results.append([name, acc, prec, rec, f1, auc])
    print(f"{name}: Acc={acc:.4f} | Prec={prec:.4f} | Rec={rec:.4f} | F1={f1:.4f} | AUC={auc:.4f}")

results_df = pd.DataFrame(results, columns=['Model','Accuracy','Precision','Recall','F1','AUC-ROC'])
results_df.to_csv('results.csv', index=False)
print("\nResults saved to results.csv")
print("\nDONE!")
import os
os.makedirs('figures', exist_ok=True)

# 1. Confusion Matrices
from sklearn.metrics import ConfusionMatrixDisplay
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
for ax, (name, (pred, prob)) in zip(axes.flatten(), models.items()):
    cm = confusion_matrix(y_val, pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=['No Disease', 'Disease'])
    disp.plot(ax=ax, colorbar=False)
    ax.set_title(name)
plt.tight_layout()
plt.savefig('figures/confusion_matrices.png', dpi=150)
print("Saved confusion_matrices.png")

# 2. ROC Curves
from sklearn.metrics import roc_curve
plt.figure(figsize=(8, 6))
for name, (pred, prob) in models.items():
    fpr, tpr, _ = roc_curve(y_val, prob)
    auc = roc_auc_score(y_val, prob)
    plt.plot(fpr, tpr, label=f'{name} (AUC={auc:.3f})')
plt.plot([0,1],[0,1],'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves')
plt.legend()
plt.tight_layout()
plt.savefig('figures/roc_curves.png', dpi=150)
print("Saved roc_curves.png")

# 3. Feature Importance (XGBoost)
plt.figure(figsize=(8, 6))
feat_imp = pd.Series(xgb.feature_importances_, index=X.columns).sort_values(ascending=True)
feat_imp.plot(kind='barh', color='steelblue')
plt.title('XGBoost Feature Importance')
plt.tight_layout()
plt.savefig('figures/feature_importance.png', dpi=150)
print("Saved feature_importance.png")

print("\nAll figures saved!")