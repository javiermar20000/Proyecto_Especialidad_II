import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score, 
    roc_auc_score, roc_curve, auc
)
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.inspection import permutation_importance

warnings.filterwarnings("ignore")

# =============================================================================
# CARGA Y PREPARACIÓN DE DATOS
# =============================================================================

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

# =============================================================================
# FUNCIONES AUXILIARES
# =============================================================================

# Obtiene el mejor modelo que no supere la precisión máxima especificada
def obtener_mejor_modelo_restringido(modelo_grid, nombre, X_train, y_train, max_precision=0.96):
    results = modelo_grid.cv_results_
    
    if nombre in ["KNN", "Random Forest"]:
        # Filtrar configuraciones que no superen la precisión máxima
        valid_indices = [i for i, score in enumerate(results['mean_test_score']) if score <= max_precision]
        
        if len(valid_indices) == 0:
            print(f"Advertencia: Ninguna configuración de {nombre} está por debajo de {max_precision*100}%")
            print(f"Usando la configuración con menor precisión disponible")
            best_idx = np.argmin(results['mean_test_score'])
        else:
            # Ordenar por precisión y tomar el mejor
            valid_scores = [(results['mean_test_score'][i], i) for i in valid_indices]
            valid_scores.sort(reverse=True)
            best_idx = valid_scores[0][1]
        
        # Obtener los parámetros del mejor modelo válido
        best_params = results['params'][best_idx]
        best_cv_score = results['mean_test_score'][best_idx]
        
        # Crear y entrenar el modelo con esos parámetros
        if nombre == "KNN":
            best_model = KNeighborsClassifier(**best_params)
        elif nombre == "Random Forest":
            best_model = RandomForestClassifier(**best_params, random_state=42)
        
        best_model.fit(X_train, y_train)
        
        print(f"Mejor {nombre} (restringido): {best_params}")
        print(f"Precisión CV: {best_cv_score:.4f}")
        
        return best_model, best_params, best_cv_score
    
    else:
        # Para regresión logística, usar el comportamiento original
        return modelo_grid.best_estimator_, modelo_grid.best_params_, modelo_grid.best_score_

# Evalúa y muestra los mejores n modelos con restricción de precisión
def top_n_modelos_restringido(modelo_grid, nombre, X_train, y_train, X_test, y_test, n=4, max_precision=0.96):
    results = modelo_grid.cv_results_
    
    # Filtrar configuraciones que no superen la precisión máxima
    if nombre in ["KNN", "Random Forest"]:
        valid_indices = [i for i, score in enumerate(results['mean_test_score']) if score <= max_precision]
        if len(valid_indices) < n:
            print(f"Advertencia: Solo {len(valid_indices)} configuraciones de {nombre} no superan {max_precision*100}% de precisión")
            valid_indices = valid_indices if valid_indices else list(range(len(results['mean_test_score'])))
        
        # Ordenar por precisión y tomar los mejores n
        valid_scores = [(results['mean_test_score'][i], i) for i in valid_indices]
        valid_scores.sort(reverse=True)
        sorted_indices = [idx for _, idx in valid_scores[:n]]
    else:
        # Para regresión logística, usar el comportamiento original
        sorted_indices = np.argsort(results['rank_test_score'])[:n]
    
    print(f"\n\n Top {min(len(sorted_indices), n)} resultados para {nombre} (máx. {max_precision*100}% precisión):")

    for rank, idx in enumerate(sorted_indices, start=1):
        params = results['params'][idx]
        cv_precision = results['mean_test_score'][idx]

        # Según el nombre del modelo, instancia el clasificador con esos hiperparámetros
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
        print(f"Precisión CV: {cv_precision:.4f}")
        print("Matriz de confusión:\n", confusion_matrix(y_test, y_pred))
        print("Reporte de clasificación:\n", classification_report(y_test, y_pred))
        print("Precisión test:", accuracy_score(y_test, y_pred))
        print("AUC:", roc_auc_score(y_test, y_proba))

def entrada_usuario():
    """
    Función para predicción interactiva basada en entrada del usuario
    """
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

# =============================================================================
# ENTRENAMIENTO Y BÚSQUEDA DE HIPERPARÁMETROS
# =============================================================================

# 6. Buscar mejores hiperparámetros y entrenar modelos

# Logistic Regression
log_params = {'C': np.logspace(-3, 3, 10)}
log_model = GridSearchCV(LogisticRegression(random_state=42), log_params, cv=5)
log_model.fit(X_train, y_train)
log_best = log_model.best_estimator_
log_pred = log_best.predict(X_test)
log_proba = log_best.predict_proba(X_test)[:, 1]

# KNN con restricción de precisión máxima del 96%
knn_params = {'n_neighbors': list(range(1, 10))}
knn_model = GridSearchCV(KNeighborsClassifier(), knn_params, cv=30)
knn_model.fit(X_train, y_train)

# Obtener el mejor modelo KNN que no supere 96%
knn_best, knn_best_params, knn_best_cv_score = obtener_mejor_modelo_restringido(
    knn_model, "KNN", X_train, y_train, max_precision=0.96)
knn_pred = knn_best.predict(X_test)
knn_proba = knn_best.predict_proba(X_test)[:, 1]

# Random Forest con restricción de precisión máxima del 96%
rf_params = {'max_depth': list(range(1, 10))}
rf_model = GridSearchCV(RandomForestClassifier(random_state=42), rf_params, cv=30)
rf_model.fit(X_train, y_train)

# Obtener el mejor modelo Random Forest que no supere 96%
rf_best, rf_best_params, rf_best_cv_score = obtener_mejor_modelo_restringido(
    rf_model, "Random Forest", X_train, y_train, max_precision=0.96)
rf_pred = rf_best.predict(X_test)
rf_proba = rf_best.predict_proba(X_test)[:, 1]

# =============================================================================
# EVALUACIÓN DE MODELOS
# =============================================================================

# 7. Evaluar múltiples configuraciones con restricción de precisión
top_n_modelos_restringido(log_model, "Regresión Logística", X_train, y_train, X_test, y_test, n=4)
top_n_modelos_restringido(knn_model, "KNN", X_train, y_train, X_test, y_test, n=4, max_precision=0.96)
top_n_modelos_restringido(rf_model, "Random Forest", X_train, y_train, X_test, y_test, n=4, max_precision=0.96)

# Preparar probabilidades para clase 0
log_proba_0 = log_best.predict_proba(X_test)[:, 0]
knn_proba_0 = knn_best.predict_proba(X_test)[:, 0]
rf_proba_0 = rf_best.predict_proba(X_test)[:, 0]

# Invertir las etiquetas: ahora la clase "0" es la positiva
y_test_invertido = 1 - y_test

# =============================================================================
# VISUALIZACIONES
# =============================================================================

# 8. Graficar curvas ROC para la clase 0
plt.figure(figsize=(8, 6))
for nombre, y_proba in zip(["Logistic Regression", "KNN", "Random Forest"], 
                          [log_proba_0, knn_proba_0, rf_proba_0]):
    fpr, tpr, _ = roc_curve(y_test_invertido, y_proba)
    plt.plot(fpr, tpr, label=f'{nombre} (AUC: {int(roc_auc_score(y_test_invertido, y_proba) * 100) / 100:.2f})')

plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('FPR (Clase 0)')
plt.ylabel('TPR (Clase 0)')
plt.title('Curvas ROC [Clase 0]')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 8.1. Graficar curvas ROC para la clase 1
plt.figure(figsize=(8, 6))
for nombre, y_proba in zip(["Logistic Regression", "KNN", "Random Forest"], 
                          [log_proba, knn_proba, rf_proba]):
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.plot(fpr, tpr, label=f'{nombre} (AUC: {int(roc_auc_score(y_test, y_proba) * 100) / 100:.2f})')

plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('FPR (Clase 1)')
plt.ylabel('TPR (Clase 1)')
plt.title('Curvas ROC [Clase 1]')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 8.2 Curvas ROC para KNN (mejores modelos)
fpr_knn, tpr_knn, _ = roc_curve(y_test, knn_proba)
roc_auc_knn = auc(fpr_knn, tpr_knn)

fpr_knn_0, tpr_knn_0, _ = roc_curve(y_test_invertido, knn_proba_0)
roc_auc_knn_0 = auc(fpr_knn_0, tpr_knn_0)

plt.figure(figsize=(8, 6))
plt.plot(fpr_knn, tpr_knn, label=f'Clase 1 (AUC = {roc_auc_knn:.2f})')
plt.plot(fpr_knn_0, tpr_knn_0, label=f'Clase 0 (AUC = {roc_auc_knn_0:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title(f'Curvas ROC del mejor modelo KNN (Restricción 96%) - {knn_best_params}')
plt.legend(loc='lower right')
plt.grid(True)
plt.tight_layout()
plt.show()

# 8.3 Curvas ROC para Random Forest (mejores modelos)
fpr_rf, tpr_rf, _ = roc_curve(y_test, rf_proba)
roc_auc_rf = auc(fpr_rf, tpr_rf)

fpr_rf_0, tpr_rf_0, _ = roc_curve(y_test_invertido, rf_proba_0)
roc_auc_rf_0 = auc(fpr_rf_0, tpr_rf_0)

plt.figure(figsize=(8, 6))
plt.plot(fpr_rf, tpr_rf, label=f'Clase 1 (AUC = {roc_auc_rf:.2f})')
plt.plot(fpr_rf_0, tpr_rf_0, label=f'Clase 0 (AUC = {roc_auc_rf_0:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title(f'Curvas ROC del mejor modelo Random Forest (Restricción 96%) - {rf_best_params}')
plt.legend(loc='lower right')
plt.grid(True)
plt.tight_layout()
plt.show()

# 8.4 Curvas ROC para Regresión Logística
fpr_log, tpr_log, _ = roc_curve(y_test, log_proba)
roc_auc_log = auc(fpr_log, tpr_log)

fpr_log_0, tpr_log_0, _ = roc_curve(y_test_invertido, log_proba_0)
roc_auc_log_0 = auc(fpr_log_0, tpr_log_0)

plt.figure(figsize=(8, 6))
plt.plot(fpr_log, tpr_log, label=f'Clase 1 (AUC = {roc_auc_log:.2f})')
plt.plot(fpr_log_0, tpr_log_0, label=f'Clase 0 (AUC = {roc_auc_log_0:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('Curvas ROC del mejor modelo Regresión Logística')
plt.legend(loc='lower right')
plt.grid(True)
plt.tight_layout()
plt.show()

# 9. Matrices de confusión heatmap
modelos = [("Logistic Regression", log_pred), ("KNN (≤96%)", knn_pred), ("Random Forest (≤96%)", rf_pred)]
for nombre, pred in modelos:
    plt.figure()
    sns.heatmap(confusion_matrix(y_test, pred), annot=True, fmt='d', cmap='Blues')
    plt.title(f'Matriz de Confusión - {nombre}')
    plt.xlabel('Predicción')
    plt.ylabel('Real')
    plt.tight_layout()
    plt.show()

# 10. Curvas de precisión vs hiperparámetros

# KNN - Mostrar solo configuraciones que no superen 96%
plt.figure()
knn_scores = knn_model.cv_results_['mean_test_score']
knn_params_values = knn_params['n_neighbors']

# Filtrar valores que no superen 96%
knn_filtered_scores = []
knn_filtered_params = []
for i, (param, score) in enumerate(zip(knn_params_values, knn_scores)):
    if score <= 0.96:
        knn_filtered_params.append(param)
        knn_filtered_scores.append(score)

plt.plot(knn_filtered_params, knn_filtered_scores, marker='o', label='≤96% precisión')
plt.plot(knn_params_values, knn_scores, marker='x', alpha=0.5, label='Todos los valores')

# Marcar el mejor modelo seleccionado
if knn_best_params['n_neighbors'] in knn_filtered_params:
    best_score = knn_scores[knn_params_values.index(knn_best_params['n_neighbors'])]
    plt.plot(knn_best_params['n_neighbors'], best_score, marker='*', markersize=15, 
             color='red', label=f'Mejor seleccionado (k={knn_best_params["n_neighbors"]})')

plt.title("Precisión KNN vs K (con restricción 96%)")
plt.xlabel("Valores de K")
plt.ylabel("Precisión")
plt.axhline(y=0.96, color='r', linestyle='--', label='Límite 96%')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Random Forest - Mostrar solo configuraciones que no superen 96%
plt.figure()
rf_scores = rf_model.cv_results_['mean_test_score']
rf_params_values = rf_params['max_depth']

# Filtrar valores que no superen 96%
rf_filtered_scores = []
rf_filtered_params = []
for i, (param, score) in enumerate(zip(rf_params_values, rf_scores)):
    if score <= 0.96:
        rf_filtered_params.append(param)
        rf_filtered_scores.append(score)

plt.plot(rf_filtered_params, rf_filtered_scores, marker='o', color='green', label='≤96% precisión')
plt.plot(rf_params_values, rf_scores, marker='x', color='green', alpha=0.5, label='Todos los valores')

# Marcar el mejor modelo seleccionado
if rf_best_params['max_depth'] in rf_filtered_params:
    best_score = rf_scores[rf_params_values.index(rf_best_params['max_depth'])]
    plt.plot(rf_best_params['max_depth'], best_score, marker='*', markersize=15, 
             color='red', label=f'Mejor seleccionado (depth={rf_best_params["max_depth"]})')

plt.title("Precisión RF vs Max Depth (con restricción 96%)")
plt.xlabel("Profundidad máxima")
plt.ylabel("Precisión")
plt.axhline(y=0.96, color='r', linestyle='--', label='Límite 96%')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Regresión Logística (sin restricción)
accuracies_test = []
for c in log_params['C']:
    model = LogisticRegression(C=c, max_iter=1000)
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    accuracies_test.append(acc)

plt.figure()
plt.plot(log_params['C'], accuracies_test, marker='o', color='red')
plt.xscale('log')
plt.title("Precisión Regresión Logística vs C")
plt.xlabel("Valores de C (log)")
plt.ylabel("Precisión")
plt.grid(True)
plt.tight_layout()
plt.show()

# =============================================================================
# GUARDADO DE MODELOS Y ANÁLISIS FINAL
# =============================================================================

# 11. Guardar mejor modelo y scaler
joblib.dump(rf_best, 'rf_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

print(f"\n✅ Modelo Random Forest guardado con restricción 96%:")
print(f"Parámetros: {rf_best_params}")
print(f"Precisión CV: {rf_best_cv_score:.4f}")
print(f"Precisión Test: {accuracy_score(y_test, rf_pred):.4f}")

# 12. Predicción interactiva (descomenta para usar)
# entrada_usuario()

# =============================================================================
# ANÁLISIS DE IMPORTANCIA DE CARACTERÍSTICAS
# =============================================================================

print("\n" + "="*60)
print("📊 ANÁLISIS DE IMPORTANCIA DE CARACTERÍSTICAS")
print("="*60)

print("\n7️⃣ CORRELACIÓN ABSOLUTA CON LA VARIABLE OBJETIVO:")
print("-" * 50)

# Calcular correlación absoluta con la variable objetivo
correlations = X.corrwith(y).abs().sort_values(ascending=False)
correlation_df = pd.DataFrame({
    'Característica': correlations.index,
    'Correlación_Abs': correlations.values,
    'Porcentaje_Corr': (correlations.values / correlations.sum()) * 100
})

# Mostrar correlaciones con target
print("Correlación absoluta con variable objetivo:")
for i, row in correlation_df.iterrows():
    print(f"{i+1:2d}. {row['Característica']:12s} - {row['Correlación_Abs']:.3f} ({row['Porcentaje_Corr']:5.2f}%)")

print("\n" + "="*60)
print("✅ ANÁLISIS COMPLETADO")
print("="*60)