import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings("ignore")

# 1. Cargar dataset
df = pd.read_csv('heart_attack_desease.csv')

# 2. Verificar y limpiar
print("Valores nulos por columna:\n", df.isnull().sum())

# 3. Separar características y etiqueta
X = df.drop('target', axis=1)
y = df['target']

# 4. Normalizar los datos
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4.5 Aplicar PCA para encontrar componentes más importantes
pca_temp = PCA(n_components=len(X.columns))
X_pca_temp = pca_temp.fit_transform(X_scaled)
componentes = pd.DataFrame(pca_temp.components_, columns=X.columns)
importancia_variables = componentes.sum(axis=0).sort_values(ascending=False)
mejores_caracteristicas = importancia_variables.index[:6].tolist()

print("Preguntas seleccionadas:\n", mejores_caracteristicas)

# Redefinir X con las mejores características
X = df[mejores_caracteristicas]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Aplicar PCA nuevamente
pca = PCA(n_components=6, random_state=42)
X_pca = pca.fit_transform(X_scaled)

# 5. Generar desbalance creando más pacientes sanos (clase 0)
original_class_counts = y.value_counts()
print("\nCantidad original por clase:\n", original_class_counts)

# Queremos aumentar los pacientes sanos (clase 0) a 600
smote = SMOTE(sampling_strategy={0: 600}, random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_pca, y)

print("\nCantidad después de SMOTE:\n", pd.Series(y_resampled).value_counts())

# 6. Dividir en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=24, stratify=y_resampled)

# 7. Modelo
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=4,
    max_features='sqrt',
    oob_score=True,
    class_weight='balanced',
    random_state=42
)

# 8. Validación cruzada
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X_train, y_train, cv=skf, scoring='accuracy')
print("Puntajes de validación cruzada:", cv_scores)
print("Promedio CV:", np.mean(cv_scores))

# 9. Entrenamiento
model.fit(X_train, y_train)

# 10. Evaluación
y_pred = model.predict(X_test)
print("\nMatriz de confusión:\n", confusion_matrix(y_test, y_pred))
print("\nReporte de clasificación:\n", classification_report(y_test, y_pred))
print("Precisión del modelo:", accuracy_score(y_test, y_pred))
print("AUC Score:", roc_auc_score(y_test, y_pred))

# 11. Guardar modelo, scaler y PCA
joblib.dump(model, 'rf_heart_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(pca, 'pca.pkl')
joblib.dump(mejores_caracteristicas, 'preguntas.pkl')  # Guardamos las preguntas seleccionadas

# 12. Predicción interactiva
def entrada_usuario():
    print("\n🔎 Responde las siguientes preguntas con datos reales:")

    preguntas_simplificadas = {
        "fbs": "¿Nivel de azúcar en sangre en ayunas > 120 mg/dl? (1: Sí, 0: No):",
        "trestbps": "Presión arterial distolica en reposo (mm Hg):",
        "ca": "Número de vasos coronarios principales coloreados afectados (0–3):",
        "chol": "Colesterol sérico (mg/dl):",
        "exang": "¿Tuvo angina inducida por el ejercicio? (1: Sí, 0: No):",
        "cp": "Tipo de dolor en el pecho (0: Angina típica, 1: Angina atípica, 2: Dolor punzante, 3: Sin dolor):"
    }

    # Cargar variables seleccionadas
    preguntas_seleccionadas = joblib.load('preguntas.pkl')

    valores = []
    for var in preguntas_seleccionadas:
        pregunta = preguntas_simplificadas.get(var)
        if pregunta:
            valor = float(input(pregunta))
            valores.append(valor)
        else:
            raise ValueError(f"Falta una pregunta para la variable '{var}'")

    # Cargar modelo, scaler y PCA
    model = joblib.load('rf_heart_model.pkl')
    scaler = joblib.load('scaler.pkl')
    pca = joblib.load('pca.pkl')

    # Procesar entrada
    entrada_np = np.array(valores).reshape(1, -1)
    entrada_scaled = scaler.transform(entrada_np)
    entrada_pca = pca.transform(entrada_scaled)

    # Predecir
    proba = model.predict_proba(entrada_pca)[0][1]  # Probabilidad de enfermo
    if proba > 0.6:
        resultado = 1
    else:
        resultado = 0

    if resultado == 1:
        print("\n🔴 Posible enfermedad cardíaca detectada.")
        print("➡️ Recomendaciones:")
        print("- Agenda una cita con un cardiólogo lo antes posible.")
        print("- Mantén una dieta saludable para el corazón.")
        print("- Realiza actividad física regularmente.")
        print("- Controla tus niveles de colesterol y presión arterial.")
    else:
        print("\n🟢 No hay signos de enfermedad cardíaca según el modelo.")
        print("➡️ Sigue manteniendo hábitos saludables.")

# 13. Ejecutar predicción si se desea
if __name__ == "__main__":
    entrada_usuario()