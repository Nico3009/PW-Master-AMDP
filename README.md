# PW-Master-AMDP

### 1️⃣ IMPORTIAMO LE LIBRERIE ###
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, balanced_accuracy_score, f1_score, roc_auc_score, confusion_matrix

# Modelli di Machine Learning
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import LocalOutlierFactor

### 2️⃣ CARICHIAMO E CAMPIONIAMO I DATI ###

# Installiamo kagglehub se non è già installato
!pip install kagglehub -q

import kagglehub
import pandas as pd
from sklearn.model_selection import train_test_split

# Scarichiamo il dataset usando kagglehub
path = kagglehub.dataset_download("ealaxi/paysim1")
print("Path to dataset files:", path)

# Cerchiamo il file CSV nel percorso scaricato
import os
import glob
csv_files = glob.glob(os.path.join(path, "*.csv"))
file_path = csv_files[0]  # Prendiamo il primo file CSV trovato
print(f"Usando il file: {file_path}")

# Carichiamo il dataset
df = pd.read_csv(file_path, sep=",", encoding="utf-8", on_bad_lines="skip")

# Visualizziamo la dimensione del dataset originale
print(f"Dimensione originale dataset: {df.shape[0]} righe")

# Campionamento stratificato di 80.000 righe
small_sample_size = 80000
# Calcoliamo la percentuale rispetto al dataset originale
sample_percentage = small_sample_size / df.shape[0]
print(f"Campionamento del {sample_percentage:.5f}% del dataset originale")

df_sample, _ = train_test_split(df, 
                               test_size=(1-sample_percentage), 
                               stratify=df["isFraud"], 
                               random_state=42)
df = df_sample.copy()
print(f"Dimensione campione: {df.shape[0]} righe")

# Rimuoviamo colonne inutili
df = df.drop(columns=["nameOrig", "nameDest"]) 

# Convertiamo la colonna categorica 'type' in numeri
encoder = LabelEncoder()
df["type"] = encoder.fit_transform(df["type"])

# Separiamo feature e target
X = df.drop(columns=["isFraud"])
y = df["isFraud"]

### 3️⃣ DIVIDIAMO I DATI E BILANCIAMO CON SMOTE ###
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

scaler = StandardScaler()
X_train_balanced = scaler.fit_transform(X_train_balanced)
X_test = scaler.transform(X_test)

### 4️⃣ DEFINIAMO I MODELLI E I PARAMETRI DELLA GRID SEARCH ###
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

models = {
    "IsolationForest": {
        "model": IsolationForest(n_jobs=-1, random_state=42),
        "params": {
            "n_estimators": [50, 100],  
            "max_samples": [0.5, 0.7],  
            "contamination": [0.05, 0.1]
        }
    },
    "LocalOutlierFactor": {
        "model": LocalOutlierFactor(novelty=True),
        "params": {
            "n_neighbors": [5, 10],  
            "metric": ["euclidean", "manhattan"]
        }
    },
    "RandomForest": {
        "model": RandomForestClassifier(n_jobs=-1, random_state=42),
        "params": {
            "n_estimators": [50, 100],  
            "max_depth": [10, 20],  
            "min_samples_split": [2, 5],  
            "max_features": ["sqrt", "log2"]
        }
    },
    "DecisionTree": {
        "model": DecisionTreeClassifier(),
        "params": {
            "max_depth": [10, 20, None],  
            "min_samples_split": [2, 5],  
            "criterion": ["gini", "entropy"]
        }
    }
}

### 5️⃣ ADDESTRIAMO I MODELLI CON GRID SEARCH ###
best_models = {}

for name, config in models.items():
    print(f"🔹 Training {name}...")
    
    grid_search = GridSearchCV(config["model"], config["params"], cv=cv, scoring="f1", n_jobs=-1)
    grid_search.fit(X_train_balanced, y_train_balanced)
    best_models[name] = grid_search.best_estimator_

    print(f"✅ Best parameters for {name}: {grid_search.best_params_}")
    print(f"✅ Best F1 score: {grid_search.best_score_}\n")

### 6️⃣ TESTIAMO I MODELLI E CALCOLIAMO LE METRICHE ###
balanced_accuracies = {}
f1_scores = {}
roc_aucs = {}
conf_matrices = {}

for name, model in best_models.items():
    print(f"📊 Evaluating {name} on test set...")
    
    if name in ["IsolationForest", "LocalOutlierFactor"]:
        y_pred = model.predict(X_test)
        y_pred = np.where(y_pred == -1, 1, 0)  
    else:
        y_pred = model.predict(X_test)

    bal_acc = balanced_accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred)
    
    balanced_accuracies[name] = bal_acc
    f1_scores[name] = f1
    roc_aucs[name] = roc_auc
    conf_matrices[name] = confusion_matrix(y_test, y_pred)

    print(f"🔹 Balanced Accuracy: {bal_acc:.4f}")
    print(f"🔹 F1 Score: {f1:.4f}")
    print(f"🔹 ROC-AUC: {roc_auc:.4f}\n")
    print(classification_report(y_test, y_pred))
    print("="*50 + "\n")

### 7️⃣ GRAFICO DELLE ACCURATEZZE BILANCIATE ###
plt.figure(figsize=(10,6))
plt.bar(balanced_accuracies.keys(), balanced_accuracies.values(), color=["blue", "green", "red", "purple"])
plt.xlabel("Modelli")
plt.ylabel("Accuratezza Bilanciata")
plt.title("Confronto delle Accuratezze Bilanciate tra i Modelli")
plt.ylim(0,1)
plt.show()

### 8️⃣ TUTTE E 4 LE MATRICI DI CONFUSIONE INSIEME ###
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

for ax, (name, conf_matrix) in zip(axes.flatten(), conf_matrices.items()):
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["No Fraud", "Fraud"], yticklabels=["No Fraud", "Fraud"], ax=ax)
    ax.set_title(f"Confusion Matrix - {name}")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

plt.tight_layout()
plt.show()

import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 🔁 Ricostruiamo X_test_df con le stesse colonne del training
feature_names = X.columns.tolist() 
X_test_df = pd.DataFrame(X_test_orig, columns=feature_names).reset_index(drop=True)
y_test_reset = pd.Series(y_test_orig).reset_index(drop=True)
feature_names = X_test_df.columns.tolist()

# ⚠️ Usa TreeExplainer per modelli tree-based
explainer = shap.TreeExplainer(model)

# ⏱️ Calcolo completo
shap_values_all = explainer.shap_values(X_test_df)

# 🎯 Prendiamo SOLO la classe positiva (frode)
shap_values = shap_values_all[:, :, 1]  # shape: (n_obs, n_features)

# 🔍 Verifica
print("✔️ shap_values shape:", shap_values.shape)
print("✔️ X_test_df shape:", X_test_df.shape)
# 🔵 Summary Plot (dot) - distribuzione e impatto
shap.summary_plot(shap_values, X_test_df, feature_names=feature_names, plot_type="dot")
plt.show()

# 📊 Summary Plot (bar) - importanza media
shap.summary_plot(shap_values, X_test_df, feature_names=feature_names, plot_type="bar")
plt.show()
from shap import Explanation

try:
    # Trova il primo indice di una frode vera
    outlier_index = y_test_reset[y_test_reset == 1].index[0]
    x_instance = X_test_df.iloc[outlier_index]
    shap_value_instance = shap_values[outlier_index]

    exp = Explanation(
        values=shap_value_instance,
        base_values=explainer.expected_value[1],
        data=x_instance.values,
        feature_names=feature_names
    )

    shap.plots.waterfall(exp)
    plt.show()

except IndexError:
    print("⚠️ Nessuna transazione fraudolenta trovata nel test set.")
# Scegli le feature più importanti (top 3)
top_features = np.argsort(np.abs(shap_values).mean(0))[::-1][:3]
top_feature_names = X_test_df.columns[top_features].tolist()

for feature in top_feature_names:
    print(f"📈 Dependence Plot per: {feature}")
    shap.dependence_plot(feature, shap_values, X_test_df, show=False)
    plt.tight_layout()
    plt.show()

