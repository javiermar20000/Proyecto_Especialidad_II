import pandas as pd  # Importa pandas para manipular estructuras de datos tipo DataFrame.
import numpy as np  # Importa numpy para cálculos numéricos.
import joblib  # Importa joblib para guardar y cargar modelos entrenados.
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score  # Para dividir datos y validarlos.
from sklearn.ensemble import RandomForestClassifier  # Importa el modelo Random Forest.
from sklearn.linear_model import LogisticRegression  # Importa el modelo de Regresión Logística.
from sklearn.neighbors import KNeighborsClassifier  # Importa el clasificador KNN.
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score  # Métricas de evaluación.
from sklearn.preprocessing import StandardScaler  # Para escalar los datos.
import warnings  # Manejo de advertencias.
warnings.filterwarnings("ignore")  # Ignora advertencias para evitar mensajes innecesarios.

# 1. Cargar el dataset
df = pd.read_csv('heart_attack_desease.csv')  # Carga el CSV en un DataFrame.

# 2. Revisar valores nulos
print("Valores nulos por columna:\n", df.isnull().sum())  # Muestra cuántos nulos hay por columna.

# 3. Separar características (X) y etiqueta (y)
X = df.drop('target', axis=1)  # Variables predictoras.
y = df['target']  # Variable objetivo.

# 4. Escalar los datos
scaler = StandardScaler()  # Instancia para estandarizar.
X_scaled = scaler.fit_transform(X)  # Escala los datos con media 0 y varianza 1.

# 5. Dividir en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(  # Divide datos en entrenamiento y prueba.
    X_scaled, y, test_size=0.2, random_state=42, stratify=y)  # Usa estratificación para mantener proporciones.

# 6. Entrenar y evaluar modelos

# --- Modelo 1: Regresión Logística ---
log_model = LogisticRegression(random_state=42)  # Instancia del modelo.
log_model.fit(X_train, y_train)  # Entrenamiento.
log_pred = log_model.predict(X_test)  # Predicciones.
print("\n--- Regresión Logística ---")
print("Matriz de confusión:\n", confusion_matrix(y_test, log_pred))  # Matriz de confusión.
print("Reporte de clasificación:\n", classification_report(y_test, log_pred))  # Métricas detalladas.
print("Precisión:", accuracy_score(y_test, log_pred))  # Precisión.
print("AUC:", roc_auc_score(y_test, log_pred))  # AUC score.

# --- Modelo 2: KNN ---
knn_model = KNeighborsClassifier(n_neighbors=5)  # Instancia del modelo KNN.
knn_model.fit(X_train, y_train)  # Entrenamiento.
knn_pred = knn_model.predict(X_test)  # Predicciones.
print("\n--- KNN ---")
print("Matriz de confusión:\n", confusion_matrix(y_test, knn_pred))
print("Reporte de clasificación:\n", classification_report(y_test, knn_pred))
print("Precisión:", accuracy_score(y_test, knn_pred))
print("AUC:", roc_auc_score(y_test, knn_pred))

# --- Modelo 3: Random Forest ---
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)  # Random Forest.
rf_model.fit(X_train, y_train)  # Entrena el modelo.
rf_pred = rf_model.predict(X_test)  # Predicciones.
print("\n--- Random Forest ---")
print("Matriz de confusión:\n", confusion_matrix(y_test, rf_pred))
print("Reporte de clasificación:\n", classification_report(y_test, rf_pred))
print("Precisión:", accuracy_score(y_test, rf_pred))
print("AUC:", roc_auc_score(y_test, rf_pred))

# 7. Guardar el modelo y el scaler
joblib.dump(rf_model, 'rf_model.pkl')  # Guarda el modelo RF.
joblib.dump(scaler, 'scaler.pkl')  # Guarda el scaler.

# 8. Predicción interactiva por usuario
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

    valores = []  # Lista para respuestas del usuario.
    for var, pregunta in preguntas.items():  # Itera por cada variable y pregunta.
        valor = float(input(pregunta))  # Solicita valor numérico.
        valores.append(valor)  # Añade a la lista.

    entrada_np = np.array(valores).reshape(1, -1)  # Convierte a array numpy.
    scaler = joblib.load('scaler.pkl')  # Carga el scaler.
    modelo = joblib.load('rf_model.pkl')  # Carga el modelo entrenado.

    entrada_scaled = scaler.transform(entrada_np)  # Escala la entrada del usuario.
    proba = modelo.predict_proba(entrada_scaled)[0][1]  # Obtiene probabilidad de clase 1.

    if proba > 0.6:  # Si la probabilidad supera 60%, se considera alto riesgo.
        print("\n🔴 Riesgo alto de enfermedad cardíaca.")
    else:
        print("\n🟢 Bajo riesgo de enfermedad cardíaca.")

# Descomenta esta línea para ejecutar la función interactiva
entrada_usuario()