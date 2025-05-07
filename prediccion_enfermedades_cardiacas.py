import pandas as pd  # Importa la librería pandas, utilizada para manipulación y análisis de datos estructurados (DataFrames).
import numpy as np  # Importa numpy, una librería para cálculos numéricos y operaciones con arrays.
import joblib  # Importa joblib, útil para guardar y cargar modelos entrenados u otros objetos grandes en Python.
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score  # Importa funciones para dividir datos y validar modelos de forma estratificada.
from sklearn.ensemble import RandomForestClassifier  # Importa el clasificador Random Forest, basado en múltiples árboles de decisión.
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score  # Importa métricas de evaluación para modelos de clasificación.
from sklearn.preprocessing import StandardScaler  # Importa StandardScaler para estandarizar las características (media 0, varianza 1).
from sklearn.decomposition import PCA  # Importa PCA para reducir la dimensionalidad y capturar la varianza principal de los datos.
from imblearn.over_sampling import SMOTE  # Importa SMOTE, técnica de sobremuestreo para balancear conjuntos de datos desbalanceados.
import warnings  # Importa warnings para manejar mensajes de advertencia.
warnings.filterwarnings("ignore")  # Oculta todas las advertencias para mantener la salida limpia.

# 1. Cargar dataset
df = pd.read_csv('heart_attack_desease.csv')  # Carga el archivo CSV en un DataFrame de pandas.

# 2. Verificar y limpiar
print("Valores nulos por columna:\n", df.isnull().sum())  # Muestra la cantidad de valores nulos por columna.

# 3. Separar características y etiqueta
X = df.drop('target', axis=1)  # Separa las características independientes, eliminando la columna 'target'.
y = df['target']  # Asigna la columna 'target' como variable dependiente (etiqueta).

# 4. Normalizar los datos
scaler = StandardScaler()  # Crea una instancia de StandardScaler.
X_scaled = scaler.fit_transform(X)  # Ajusta y transforma X para que tenga media 0 y desviación estándar 1.

# 4.5 Aplicar PCA para encontrar componentes más importantes
pca_temp = PCA(n_components=len(X.columns))  # Crea un PCA temporal con tantos componentes como columnas originales.
X_pca_temp = pca_temp.fit_transform(X_scaled)  # Ajusta y transforma los datos normalizados con PCA.
componentes = pd.DataFrame(pca_temp.components_, columns=X.columns)  # Crea un DataFrame con los pesos de cada componente.
importancia_variables = componentes.sum(axis=0).sort_values(ascending=False)  # Suma los pesos por variable y los ordena por importancia.
mejores_caracteristicas = importancia_variables.index[:6].tolist()  # Selecciona las 6 características con mayor contribución.

print("Preguntas seleccionadas:\n", mejores_caracteristicas)  # Imprime las características seleccionadas.

# Redefinir X con las mejores características
X = df[mejores_caracteristicas]  # Redefine X usando solo las características seleccionadas.
scaler = StandardScaler()  # Crea una nueva instancia de StandardScaler.
X_scaled = scaler.fit_transform(X)  # Estandariza los datos nuevamente con las nuevas características.

# Aplicar PCA nuevamente
pca = PCA(n_components=6, random_state=42)  # Crea una instancia de PCA con 6 componentes principales.
X_pca = pca.fit_transform(X_scaled)  # Ajusta y transforma los datos seleccionados con PCA.

# 5. Generar desbalance creando más pacientes sanos (clase 0)
original_class_counts = y.value_counts()  # Cuenta cuántas muestras hay por clase antes del balanceo.
print("\nCantidad original por clase:\n", original_class_counts)  # Imprime la cantidad original por clase.

# Queremos aumentar los pacientes sanos (clase 0) a 600
smote = SMOTE(sampling_strategy={0: 600}, random_state=42)  # Crea una instancia de SMOTE para generar muestras sintéticas de la clase 0.
X_resampled, y_resampled = smote.fit_resample(X_pca, y)  # Aplica SMOTE al conjunto PCA, generando un conjunto balanceado.

print("\nCantidad después de SMOTE:\n", pd.Series(y_resampled).value_counts())  # Imprime la cantidad de cada clase después del balanceo.

# 6. Dividir en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(  # Divide el conjunto balanceado en entrenamiento y prueba.
    X_resampled, y_resampled, test_size=0.2, random_state=24, stratify=y_resampled)  # Usa muestreo estratificado para conservar proporciones de clase.

# 7. Modelo
model = RandomForestClassifier(  # Crea un clasificador Random Forest con hiperparámetros personalizados.
    n_estimators=200,  # Número de árboles en el bosque.
    max_depth=10,  # Profundidad máxima de los árboles.
    min_samples_split=5,  # Mínimo de muestras para dividir un nodo.
    min_samples_leaf=4,  # Mínimo de muestras en una hoja.
    max_features='sqrt',  # Usa la raíz cuadrada del número de características para buscar la mejor división.
    oob_score=True,  # Habilita el cálculo del Out-of-Bag score para validación interna.
    class_weight='balanced',  # Ajusta los pesos de clase automáticamente según su frecuencia.
    random_state=42  # Fija la semilla para reproducibilidad.
)

# 8. Validación cruzada
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)  # Define un validador cruzado estratificado con 5 particiones y barajado.
cv_scores = cross_val_score(model, X_train, y_train, cv=skf, scoring='accuracy')  # Ejecuta validación cruzada con métrica de precisión.
print("Puntajes de validación cruzada:", cv_scores)  # Muestra los puntajes de precisión por cada partición.
print("Promedio CV:", np.mean(cv_scores))  # Muestra el promedio de los puntajes (precisión media).

# 9. Entrenamiento
model.fit(X_train, y_train)  # Entrena el modelo con los datos de entrenamiento.

# 10. Evaluación
y_pred = model.predict(X_test)  # Predice las etiquetas para el conjunto de prueba.
print("\nMatriz de confusión:\n", confusion_matrix(y_test, y_pred))  # Muestra la matriz de confusión.
print("\nReporte de clasificación:\n", classification_report(y_test, y_pred))  # Muestra métricas como precisión, recall y F1.
print("Precisión del modelo:", accuracy_score(y_test, y_pred))  # Muestra la precisión total del modelo.
print("AUC Score:", roc_auc_score(y_test, y_pred))  # Muestra el AUC, útil para evaluar el rendimiento con clases desbalanceadas.

# 11. Guardar modelo, scaler y PCA
joblib.dump(model, 'rf_heart_model.pkl')  # Guarda el modelo entrenado en un archivo.
joblib.dump(scaler, 'scaler.pkl')  # Guarda el objeto de escalado para reutilizar en predicciones futuras.
joblib.dump(pca, 'pca.pkl')  # Guarda el modelo PCA entrenado.
joblib.dump(mejores_caracteristicas, 'preguntas.pkl')  # Guarda las variables seleccionadas como importantes.

# 12. Predicción interactiva
def entrada_usuario():  # Define una función para ingresar datos de usuario y predecir el riesgo.
    print("\n🔎 Responde las siguientes preguntas con datos reales:")

    preguntas_simplificadas = {  # Diccionario de preguntas simplificadas para cada variable seleccionada.
        "fbs": "¿Nivel de azúcar en sangre en ayunas > 120 mg/dl? (1: Sí, 0: No):",
        "trestbps": "Presión arterial distolica en reposo (mm Hg):",
        "ca": "Número de vasos coronarios principales coloreados afectados (0–3):",
        "chol": "Colesterol sérico (mg/dl):",
        "exang": "¿Tuvo angina inducida por el ejercicio? (1: Sí, 0: No):",
        "cp": "Tipo de dolor en el pecho (0: Angina típica, 1: Angina atípica, 2: Dolor punzante, 3: Sin dolor):"
    }

    preguntas_seleccionadas = joblib.load('preguntas.pkl')  # Carga las variables seleccionadas previamente.

    valores = []  # Lista para almacenar las respuestas del usuario.
    for var in preguntas_seleccionadas:  # Itera sobre cada variable seleccionada.
        pregunta = preguntas_simplificadas.get(var)  # Obtiene la pregunta correspondiente.
        if pregunta:
            valor = float(input(pregunta))  # Solicita al usuario ingresar un valor numérico.
            valores.append(valor)  # Agrega el valor a la lista.
        else:
            raise ValueError(f"Falta una pregunta para la variable '{var}'")  # Error si no se encuentra la pregunta.

    # Cargar modelo, scaler y PCA
    model = joblib.load('rf_heart_model.pkl')  # Carga el modelo entrenado.
    scaler = joblib.load('scaler.pkl')  # Carga el escalador.
    pca = joblib.load('pca.pkl')  # Carga el modelo PCA.

    entrada_np = np.array(valores).reshape(1, -1)  # Convierte las respuestas a un array de NumPy con forma adecuada.
    entrada_scaled = scaler.transform(entrada_np)  # Aplica escalado a los datos del usuario.
    entrada_pca = pca.transform(entrada_scaled)  # Aplica reducción de dimensionalidad con PCA.

    proba = model.predict_proba(entrada_pca)[0][1]  # Obtiene la probabilidad de tener enfermedad (clase 1).
    if proba > 0.6:  # Si la probabilidad es mayor al 60%, se considera riesgo alto.
        resultado = 1
    else:
        resultado = 0

    if resultado == 1:  # Si el resultado indica enfermedad:
        print("\n🔴 Posible enfermedad cardíaca detectada.")
        print("➡️ Recomendaciones:")
        print("- Agenda una cita con un cardiólogo lo antes posible.")
        print("- Mantén una dieta saludable para el corazón.")
        print("- Realiza actividad física regularmente.")
        print("- Controla tus niveles de colesterol y presión arterial.")
    else:  # Si no se detecta enfermedad:
        print("\n🟢 No hay signos de enfermedad cardíaca según el modelo.")
        print("➡️ Sigue manteniendo hábitos saludables.")

# 13. Ejecutar predicción si se desea
if __name__ == "__main__":  # Si se ejecuta directamente el script...
    entrada_usuario()  # ...se llama a la función para hacer predicción con entrada del usuario.