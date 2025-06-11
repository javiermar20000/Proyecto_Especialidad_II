import pandas as pd                         # Importa la biblioteca pandas y la asigna al alias 'pd' para manipulación de datos tabulares
import numpy as np                          # Importa la biblioteca numpy con el alias 'np' para operaciones numéricas y manejo de arrays
import joblib                               # Importa joblib para guardar y cargar modelos entrenados u otros objetos de Python
import matplotlib.pyplot as plt             # Importa la biblioteca matplotlib para visualización de gráficos, asignándola al alias 'plt'
from sklearn.model_selection import train_test_split, GridSearchCV  # Importa funciones de sklearn para dividir datos en entrenamiento/prueba y para realizar búsqueda de hiperparámetros
from sklearn.ensemble import RandomForestClassifier  # Importa el clasificador Random Forest del módulo ensemble de sklearn
from sklearn.linear_model import LogisticRegression  # Importa el modelo de Regresión Logística del módulo linear_model de sklearn
from sklearn.neighbors import KNeighborsClassifier  # Importa el clasificador k-vecinos más cercanos del módulo neighbors
from sklearn.metrics import ( # Importa varias métricas de evaluación de modelos desde sklearn.metrics
    classification_report, confusion_matrix, accuracy_score, 
    roc_auc_score, roc_curve, auc
)  
from sklearn.preprocessing import StandardScaler  # Importa el escalador estándar para normalizar características (media 0, desviación estándar 1)
import seaborn as sns                        # Importa la biblioteca seaborn, útil para gráficos estadísticos, con alias 'sns'
import warnings                              # Importa el módulo warnings para gestionar advertencias del sistema
warnings.filterwarnings("ignore")           # Ignora las advertencias generadas durante la ejecución del código

# 1. Cargar el dataset
df = pd.read_csv('heart_attack_desease.csv')         # Lee el archivo CSV y lo carga en un DataFrame llamado 'df'

# 2. Revisar valores nulos
print("Valores nulos por columna:\n", df.isnull().sum())  # Muestra la cantidad de valores nulos por columna para verificar datos faltantes

# 3. Separar características (X) y etiqueta (y)
X = df.drop('target', axis=1)                        # Crea la matriz de características eliminando la columna 'target' del DataFrame
y = df['target']                                     # Define el vector objetivo (etiqueta) como la columna 'target'

# 4. Escalar los datos
scaler = StandardScaler()                            # Crea una instancia del escalador estándar para normalizar las características
X_scaled = scaler.fit_transform(X)                   # Ajusta el escalador a los datos y transforma X para que tenga media 0 y varianza 1

# 5. Dividir en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y) # Divide los datos escalados en conjuntos de entrenamiento y prueba (80/20)
# 'random_state=42' asegura reproducibilidad
# 'stratify=y' mantiene la proporción de clases en ambos conjuntos

# 6. Buscar mejores hiperparámetros y entrenar modelos

# Logistic Regression
log_params = {'C': np.logspace(-3, 3, 10)}               # Define una grilla de valores para el hiperparámetro 'C' (regularización) en una escala logarítmica
log_model = GridSearchCV(LogisticRegression(random_state=42), log_params, cv=5)  # Crea un GridSearchCV para buscar el mejor valor de 'C' usando validación cruzada de 5 pliegues
log_model.fit(X_train, y_train)                         # Ajusta el modelo de regresión logística con los mejores hiperparámetros encontrados
log_best = log_model.best_estimator_                    # Extrae el mejor modelo (estimador) después de la búsqueda
log_pred = log_best.predict(X_test)                     # Realiza predicciones de clases sobre el conjunto de prueba
log_proba = log_best.predict_proba(X_test)[:, 1]        # Calcula las probabilidades predichas de la clase positiva (1) para métricas como ROC AUC

# KNN
knn_params = {'n_neighbors': list(range(1, 10))}        # Define una grilla de valores de vecinos (de 1 a 9) para probar en el clasificador KNN
knn_model = GridSearchCV(KNeighborsClassifier(), knn_params, cv=5)  # Crea un GridSearchCV para buscar el mejor número de vecinos usando validación cruzada
knn_model.fit(X_train, y_train)                         # Entrena el modelo KNN con los mejores hiperparámetros
knn_best = knn_model.best_estimator_                    # Extrae el mejor modelo encontrado por GridSearch
knn_pred = knn_best.predict(X_test)                     # Predice las clases del conjunto de prueba usando el mejor modelo KNN
knn_proba = knn_best.predict_proba(X_test)[:, 1]        # Calcula las probabilidades predichas de la clase positiva (1)

# Random Forest
rf_params = {'max_depth': list(range(1, 10))}           # Define una grilla de valores para la profundidad máxima del árbol (1 a 9)
rf_model = GridSearchCV(RandomForestClassifier(random_state=42), rf_params, cv=5)  # Crea un GridSearchCV para ajustar el clasificador Random Forest buscando la mejor profundidad
rf_model.fit(X_train, y_train)                          # Entrena el modelo Random Forest con los mejores hiperparámetros
rf_best = rf_model.best_estimator_                      # Extrae el mejor modelo Random Forest encontrado
rf_pred = rf_best.predict(X_test)                       # Predice las clases del conjunto de prueba con el mejor modelo
rf_proba = rf_best.predict_proba(X_test)[:, 1]          # Calcula las probabilidades de la clase positiva (1) para métricas de evaluación

# 7. Evaluar múltiples configuraciones
def top_n_modelos(modelo_grid, nombre, X_train, y_train, X_test, y_test, n=4):
    results = modelo_grid.cv_results_                   # Obtiene todos los resultados de la validación cruzada del GridSearchCV
    sorted_indices = np.argsort(results['rank_test_score'])[:n]  # Ordena los resultados por el ranking del score (menor es mejor) y selecciona los índices de las top-n configuraciones
    print(f"\n\n Top {n} resultados para {nombre}:")    # Imprime el encabezado para los mejores modelos de ese tipo

    for rank, idx in enumerate(sorted_indices, start=1):  # Itera sobre los mejores modelos, obteniendo su posición (rank) e índice original en el array
        params = results['params'][idx]                 # Extrae los hiperparámetros de ese modelo específico

        # Según el nombre del modelo, instancia el clasificador con esos hiperparámetros
        if nombre == "Regresión Logística":
            modelo = LogisticRegression(**params, random_state=42)
        elif nombre == "KNN":
            modelo = KNeighborsClassifier(**params)
        elif nombre == "Random Forest":
            modelo = RandomForestClassifier(**params, random_state=42)
        else:
            continue                                    # Si el nombre no coincide, salta esa iteración

        modelo.fit(X_train, y_train)                    # Entrena el modelo con los datos de entrenamiento

        y_pred = modelo.predict(X_test)                 # Predice las clases con el conjunto de prueba
        y_proba = modelo.predict_proba(X_test)[:, 1]    # Predice las probabilidades de la clase positiva (1)

        print(f"\n--- {nombre} (Top #{rank}) ---")      # Muestra qué configuración se está evaluando
        print("Configuración:", params)                 # Muestra los hiperparámetros usados
        print("Matriz de confusión:\n", confusion_matrix(y_test, y_pred))  # Imprime la matriz de confusión entre predicciones y etiquetas reales
        print("Reporte de clasificación:\n", classification_report(y_test, y_pred))  # Imprime métricas como precisión, recall y f1-score para cada clase
        print("Precisión:", accuracy_score(y_test, y_pred))  # Muestra la precisión (accuracy) del modelo
        print("AUC:", roc_auc_score(y_test, y_proba))   # Calcula y muestra el AUC (área bajo la curva ROC)

# Evaluar los 3 mejores resultados de cada modelo
top_n_modelos(log_model, "Regresión Logística", X_train, y_train, X_test, y_test, n=3)  # Evalúa las 3 mejores configuraciones del modelo de regresión logística
top_n_modelos(knn_model, "KNN", X_train, y_train, X_test, y_test, n=3)  # Evalúa las 3 mejores configuraciones del modelo KNN
top_n_modelos(rf_model, "Random Forest", X_train, y_train, X_test, y_test, n=4)  # Evalúa las 4 mejores configuraciones del modelo Random Forest


log_proba_0 = log_best.predict_proba(X_test)[:, 0]
knn_proba_0 = knn_best.predict_proba(X_test)[:, 0]
rf_proba_0 = rf_best.predict_proba(X_test)[:, 0]

# Invertimos las etiquetas: ahora la clase "0" es la positiva
y_test_invertido = 1 - y_test

# 8. Graficar curvas ROC para la clase 0
plt.figure(figsize=(8, 6))  # Crea una figura nueva con tamaño 8x6 pulgadas
# Itera sobre cada modelo junto con sus probabilidades de clase positiva (y_proba)
for nombre, y_proba in zip(["Logistic Regression", "KNN", "Random Forest"], [log_proba_0, knn_proba_0, rf_proba_0]):  

    fpr, tpr, _ = roc_curve(y_test_invertido, y_proba)  # Calcula los valores de FPR (False Positive Rate) y TPR (True Positive Rate) para la curva ROC
    plt.plot(fpr, tpr, label=f'{nombre} (AUC: {roc_auc_score(y_test_invertido, y_proba):.2f})')  # Dibuja la curva ROC para ese modelo e incluye en la leyenda su AUC correspondiente

plt.plot([0, 1], [0, 1], 'k--')  # Dibuja una línea diagonal como referencia (clasificador aleatorio)
plt.xlabel('FPR (Clase 0)')  # Etiqueta para el eje X: Tasa de Falsos Positivos
plt.ylabel('TPR (Clase 0)')  # Etiqueta para el eje Y: Tasa de Verdaderos Positivos
plt.title('Curvas ROC [Clase 0]')  # Título del gráfico
plt.legend()  # Muestra la leyenda con los nombres de los modelos y sus AUC
plt.grid(True)  # Agrega una cuadrícula al gráfico
plt.tight_layout()  # Ajusta el diseño para evitar que se superpongan elementos
plt.show()  # Muestra el gráfico en pantalla

# 8.1. Graficar curvas ROC para la clase 1
plt.figure(figsize=(8, 6))  # Crea una figura nueva con tamaño 8x6 pulgadas
# Itera sobre cada modelo junto con sus probabilidades de clase positiva (y_proba)
for nombre, y_proba in zip(["Logistic Regression", "KNN", "Random Forest"], [log_proba, knn_proba, rf_proba]):  

    fpr, tpr, _ = roc_curve(y_test, y_proba)  # Calcula los valores de FPR (False Positive Rate) y TPR (True Positive Rate) para la curva ROC
    plt.plot(fpr, tpr, label=f'{nombre} (AUC: {roc_auc_score(y_test, y_proba):.2f})')  # Dibuja la curva ROC para ese modelo e incluye en la leyenda su AUC correspondiente

plt.plot([0, 1], [0, 1], 'k--')  # Dibuja una línea diagonal como referencia (clasificador aleatorio)
plt.xlabel('FPR (Clase 1)')  # Etiqueta para el eje X: Tasa de Falsos Positivos
plt.ylabel('TPR (Clase 1)')  # Etiqueta para el eje Y: Tasa de Verdaderos Positivos
plt.title('Curvas ROC [Clase 1]')  # Título del gráfico
plt.legend()  # Muestra la leyenda con los nombres de los modelos y sus AUC
plt.grid(True)  # Agrega una cuadrícula al gráfico
plt.tight_layout()  # Ajusta el diseño para evitar que se superpongan elementos
plt.show()  # Muestra el gráfico en pantalla

# 8.2 Curvas ROC para discriminar entre clases de los mejores modelos encontrados (KNN)
# Calcular curvas ROC
fpr_knn, tpr_knn, _ = roc_curve(y_test, knn_proba)
roc_auc_knn = auc(fpr_knn, tpr_knn)

fpr_knn_0, tpr_knn_0, _ = roc_curve(y_test_invertido, knn_proba_0)
roc_auc_knn_0 = auc(fpr_knn_0, tpr_knn_0)

# Plot
plt.figure(figsize=(8, 6))
plt.plot(fpr_knn, tpr_knn, label=f'Clase 1 (AUC = {roc_auc_knn:.2f})')
plt.plot(fpr_knn_0, tpr_knn_0, label=f'Clase 0 (AUC = {roc_auc_knn_0:.2f})')

# Diagonal
plt.plot([0, 1], [0, 1], 'k--')

plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('Curvas ROC del mejor modelo KNN')
plt.legend(loc='lower right')
plt.grid(True)
plt.tight_layout()
plt.show()

# 8.3 Curvas ROC para discriminar entre clases de los mejores modelos encontrados (Random Forest)
# Calcular curvas ROC
fpr_rf, tpr_rf, _ = roc_curve(y_test, rf_proba)
roc_auc_rf = auc(fpr_rf, tpr_rf)

fpr_rf_0, tpr_rf_0, _ = roc_curve(y_test_invertido, rf_proba_0)
roc_auc_rf_0 = auc(fpr_rf_0, tpr_rf_0)

# Plot
plt.figure(figsize=(8, 6))
plt.plot(fpr_rf, tpr_rf, label=f'Clase 1 (AUC = {roc_auc_rf:.2f})')
plt.plot(fpr_rf_0, tpr_rf_0, label=f'Clase 0 (AUC = {roc_auc_rf_0:.2f})')

# Diagonal
plt.plot([0, 1], [0, 1], 'k--')

plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('Curvas ROC del mejor modelo Random Forest')
plt.legend(loc='lower right')
plt.grid(True)
plt.tight_layout()
plt.show()

# 8.4 Curvas ROC para discriminar entre clases de los mejores modelos encontrados (Regresión logistica)
# Calcular curvas ROC
fpr_log, tpr_log, _ = roc_curve(y_test, log_proba)
roc_auc_log = auc(fpr_log, tpr_log)

fpr_log_0, tpr_log_0, _ = roc_curve(y_test_invertido, log_proba_0)
roc_auc_log_0 = auc(fpr_log_0, tpr_log_0)

# Plot
plt.figure(figsize=(8, 6))
plt.plot(fpr_log, tpr_log, label=f'Clase 1 (AUC = {roc_auc_log:.2f})')
plt.plot(fpr_log_0, tpr_log_0, label=f'Clase 0 (AUC = {roc_auc_log_0:.2f})')

# Diagonal
plt.plot([0, 1], [0, 1], 'k--')

plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('Curvas ROC del mejor modelo Regresión Lineal')
plt.legend(loc='lower right')
plt.grid(True)
plt.tight_layout()
plt.show()

# 9. Matrices de confusión heatmap
# Crea una lista de tuplas con el nombre del modelo y sus predicciones en el test set
modelos = [("Logistic Regression", log_pred), ("KNN", knn_pred), ("Random Forest", rf_pred)]
for nombre, pred in modelos: # Itera sobre cada modelo y sus predicciones
    plt.figure() # Crea una nueva figura para cada heatmap
    # Dibuja un mapa de calor de la matriz de confusión con anotaciones, usando el esquema de colores 'Blues'
    sns.heatmap(confusion_matrix(y_test, pred), annot=True, fmt='d', cmap='Blues')
    plt.title(f'Matriz de Confusión - {nombre}') # Título con el nombre del modelo
    plt.xlabel('Predicción') # Etiqueta del eje X
    plt.ylabel('Real') # Etiqueta del eje Y
    plt.tight_layout() # Ajusta automáticamente los márgenes para evitar solapamientos
    plt.show() # Muestra el gráfico

# 10. Curvas de precisión vs hiperparámetros
# KNN
plt.figure() # Crea una nueva figura
# Grafica la precisión promedio del cross-validation frente al número de vecinos (k)
plt.plot(knn_params['n_neighbors'], knn_model.cv_results_['mean_test_score'], marker='o')
plt.title("Precisión KNN vs K") # Título del gráfico
plt.xlabel("K") # Etiqueta del eje X
plt.ylabel("Precisión") # Etiqueta del eje Y
plt.grid(True) # Activa la cuadrícula
plt.tight_layout() # Ajusta el layout
plt.show() # Muestra el gráfico

# Random Forest
plt.figure()
# Grafica la precisión media frente a la profundidad máxima del árbol en Random Forest
plt.plot(rf_params['max_depth'], rf_model.cv_results_['mean_test_score'], marker='o', color='green')
plt.title("Precisión RF vs Max Depth") # Título del gráfico
plt.xlabel("Profundidad máxima") # Etiqueta del eje X
plt.ylabel("Precisión") # Etiqueta del eje Y
plt.grid(True) # Activa la cuadrícula
plt.tight_layout() # Ajusta el layout
plt.show() # Muestra el gráfico

# Logistic Regression
plt.figure()
# Grafica la precisión frente al valor de C (parámetro de regularización)
plt.plot(log_params['C'], log_model.cv_results_['mean_test_score'], marker='o', color='red')
plt.xscale('log') # Escala logarítmica en el eje X (ya que C varía exponencialmente)
plt.title("Precisión RL vs C") # Título del gráfico
plt.xlabel("C (log)") # Etiqueta X indicando escala log
plt.ylabel("Precisión") # Etiqueta del eje Y
plt.grid(True) # Activa la cuadrícula
plt.tight_layout() # Ajusta el layout
plt.show() # Muestra el gráfico

# 11. Guardar mejor modelo y scaler
joblib.dump(rf_best, 'rf_model.pkl')  # Guarda el mejor modelo de Random Forest en un archivo .pkl
joblib.dump(scaler, 'scaler.pkl')     # Guarda el escalador estándar para uso posterior (ej. predicciones nuevas)

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

    # Inicializa una lista vacía donde se guardarán los valores ingresados por el usuario
    valores = []
    for var, pregunta in preguntas.items(): # Itera sobre el diccionario 'preguntas', donde la clave es el nombre de la variable y el valor es el texto de la pregunta
        valor = float(input(pregunta)) # Muestra la pregunta al usuario y convierte la respuesta a float (valor numérico)
        valores.append(valor) # Agrega el valor ingresado a la lista 'valores'

    entrada_np = np.array(valores).reshape(1, -1) # Convierte la lista a un arreglo NumPy y lo redimensiona a una matriz de una fila (1 muestra, n columnas)
    scaler = joblib.load('scaler.pkl') # Carga el objeto de escalado previamente guardado (StandardScaler)
    modelo = joblib.load('rf_model.pkl') # Carga el modelo Random Forest entrenado y guardado previamente

    entrada_scaled = scaler.transform(entrada_np) # Escala la entrada del usuario con el mismo escalador usado durante el entrenamiento
    # Calcula la probabilidad de que la entrada pertenezca a la clase positiva (clase 1)
    # `predict_proba` devuelve una matriz con dos columnas: [proba_clase_0, proba_clase_1]
    # Por eso se accede con `[0][1]` para obtener la probabilidad de clase 1 de la primera (y única) muestra
    proba = modelo.predict_proba(entrada_scaled)[0][1]

    # Si la prob es mayor a 0.6 hay riesgo de enfermedad cardiaca
    if proba > 0.6:
        print("\n🔴 Riesgo alto de enfermedad cardíaca.")
    else: # en caso contrario la persona estara sana
        print("\n🟢 Bajo riesgo de enfermedad cardíaca.")

# Descomentar para usar en consola
# entrada_usuario()