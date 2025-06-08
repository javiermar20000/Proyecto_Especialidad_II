import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score, 
    roc_auc_score, roc_curve, precision_recall_fscore_support
)
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")

# 1. Cargar el dataset
df = pd.read_csv('heart_attack_desease.csv')

# 2. Revisar valores nulos
print("Valores nulos por columna:\n", df.isnull().sum())

# 3. Separar características (X) y etiqueta (y)
X = df.drop('target', axis=1)
y = df['target']

# 4. Escalar los datos
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 5. Dividir en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# 6. Buscar mejores hiperparámetros y entrenar modelos

# Logistic Regression
log_params = {'C': np.logspace(-3, 3, 10)}
log_model = GridSearchCV(LogisticRegression(random_state=42), log_params, cv=5)
log_model.fit(X_train, y_train)
log_best = log_model.best_estimator_
log_pred = log_best.predict(X_test)
log_proba = log_best.predict_proba(X_test)[:, 1]

# KNN
knn_params = {'n_neighbors': list(range(1, 31))}
knn_model = GridSearchCV(KNeighborsClassifier(), knn_params, cv=5)
knn_model.fit(X_train, y_train)
knn_best = knn_model.best_estimator_
knn_pred = knn_best.predict(X_test)
knn_proba = knn_best.predict_proba(X_test)[:, 1]

# Random Forest
rf_params = {'max_depth': list(range(1, 21))}
rf_model = GridSearchCV(RandomForestClassifier(random_state=42), rf_params, cv=5)
rf_model.fit(X_train, y_train)
rf_best = rf_model.best_estimator_
rf_pred = rf_best.predict(X_test)
rf_proba = rf_best.predict_proba(X_test)[:, 1]

# 7. Evaluar múltiples configuraciones
def top_n_modelos(modelo_grid, nombre, X_train, y_train, X_test, y_test, n=3):
    results = modelo_grid.cv_results_
    sorted_indices = np.argsort(results['rank_test_score'])[:n]  # top n configs

    print(f"\n\n🎯 Top {n} resultados para {nombre}:")

    for rank, idx in enumerate(sorted_indices, start=1):
        params = results['params'][idx]
        if nombre == "Regresión Logística":
            modelo = LogisticRegression(**params, random_state=42)
        elif nombre == "KNN":
            modelo = KNeighborsClassifier(**params)
        elif nombre == "Random Forest":
            modelo = RandomForestClassifier(**params, random_state=42)
        else:
            continue

        modelo.fit(X_train, y_train)
        y_pred = modelo.predict(X_test)
        y_proba = modelo.predict_proba(X_test)[:, 1]

        print(f"\n--- {nombre} (Top #{rank}) ---")
        print("Configuración:", params)
        print("Matriz de confusión:\n", confusion_matrix(y_test, y_pred))
        print("Reporte de clasificación:\n", classification_report(y_test, y_pred))
        print("Precisión:", accuracy_score(y_test, y_pred))
        print("AUC:", roc_auc_score(y_test, y_proba))

# Evaluar los 3 mejores resultados de cada modelo
top_n_modelos(log_model, "Regresión Logística", X_train, y_train, X_test, y_test, n=3)
top_n_modelos(knn_model, "KNN", X_train, y_train, X_test, y_test, n=3)
top_n_modelos(rf_model, "Random Forest", X_train, y_train, X_test, y_test, n=3)


# 8. Graficar curvas ROC
plt.figure(figsize=(8, 6))
for nombre, y_proba in zip(["Logistic Regression", "KNN", "Random Forest"],
                           [log_proba, knn_proba, rf_proba]):
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.plot(fpr, tpr, label=f'{nombre} (AUC: {roc_auc_score(y_test, y_proba):.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('Curvas ROC')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 9. Matrices de confusión heatmap
modelos = [("Logistic Regression", log_pred), ("KNN", knn_pred), ("Random Forest", rf_pred)]
for nombre, pred in modelos:
    plt.figure()
    sns.heatmap(confusion_matrix(y_test, pred), annot=True, fmt='d', cmap='Blues')
    plt.title(f'Matriz de Confusión - {nombre}')
    plt.xlabel('Predicción')
    plt.ylabel('Real')
    plt.tight_layout()
    plt.show()

# 10. Curvas de precisión vs hiperparámetros
# KNN
plt.figure()
plt.plot(knn_params['n_neighbors'], knn_model.cv_results_['mean_test_score'], marker='o')
plt.title("Precisión KNN vs K")
plt.xlabel("K")
plt.ylabel("Precisión")
plt.grid(True)
plt.tight_layout()
plt.show()

# Random Forest
plt.figure()
plt.plot(rf_params['max_depth'], rf_model.cv_results_['mean_test_score'], marker='o', color='green')
plt.title("Precisión RF vs Max Depth")
plt.xlabel("Profundidad máxima")
plt.ylabel("Precisión")
plt.grid(True)
plt.tight_layout()
plt.show()

# Logistic Regression
plt.figure()
plt.plot(log_params['C'], log_model.cv_results_['mean_test_score'], marker='o', color='red')
plt.xscale('log')
plt.title("Precisión RL vs C")
plt.xlabel("C (log)")
plt.ylabel("Precisión")
plt.grid(True)
plt.tight_layout()
plt.show()

# 11. Guardar mejor modelo y scaler
joblib.dump(rf_best, 'rf_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

# 12. Predicción interactiva
def entrada_usuario():
    print("\n🔎 Por favor, responde las siguientes preguntas:")

    preguntas = {
        "age": "Edad (años): ",
        "sex": "Sexo (1: Hombre, 0: Mujer): ",
        "cp": "Tipo de dolor en el pecho (0-3): ",
        "trestbps": "Presión arterial en reposo (mm Hg): ",
        "chol": "Colesterol sérico (mg/dl): ",
        "fbs": "Azúcar en sangre en ayunas > 120 mg/dl (1: Sí, 0: No): ",
        "restecg": "Resultados del ECG en reposo (0-2): ",
        "thalach": "Frecuencia cardíaca máxima alcanzada: ",
        "exang": "Angina inducida por ejercicio (1: Sí, 0: No): ",
        "oldpeak": "Depresión ST inducida por el ejercicio: ",
        "slope": "Pendiente del segmento ST (0-2): ",
        "ca": "Nº de vasos principales coloreados (0-3): ",
        "thal": "Resultado del test Thal (1 = normal; 2 = fijo; 3 = reversible): "
    }

    valores = []
    for var, pregunta in preguntas.items():
        valor = float(input(pregunta))
        valores.append(valor)

    entrada_np = np.array(valores).reshape(1, -1)
    scaler = joblib.load('scaler.pkl')
    modelo = joblib.load('rf_model.pkl')

    entrada_scaled = scaler.transform(entrada_np)
    proba = modelo.predict_proba(entrada_scaled)[0][1]

    if proba > 0.6:
        print("\n🔴 Riesgo alto de enfermedad cardíaca.")
    else:
        print("\n🟢 Bajo riesgo de enfermedad cardíaca.")

# Descomentar para usar en consola
# entrada_usuario()