import pandas as pd                         # Importa la biblioteca pandas para manipulación y análisis de datos (estructuras tipo DataFrame).
import numpy as np                          # Importa la biblioteca NumPy para operaciones numéricas y manejo de arrays.
import joblib                               # Importa joblib para guardar y cargar modelos entrenados u otros objetos de Python.
import matplotlib.pyplot as plt             # Importa matplotlib para generar gráficos y visualizaciones.
import seaborn as sns                       # Importa seaborn para visualizaciones estadísticas más estéticas y avanzadas.
import warnings                             # Importa el módulo warnings para controlar la aparición de advertencias.
from sklearn.model_selection import train_test_split, GridSearchCV    # Importa funciones para dividir datos y hacer búsqueda de hiperparámetros.
from sklearn.ensemble import RandomForestClassifier                   # Importa el clasificador de bosque aleatorio (Random Forest).
from sklearn.linear_model import LogisticRegression                   # Importa el modelo de regresión logística.
from sklearn.neighbors import KNeighborsClassifier                    # Importa el clasificador basado en vecinos más cercanos (KNN).
from sklearn.metrics import (                                         # Importa métricas para evaluar modelos de clasificación:
    classification_report,                                            # - Reporte con precisión, recall, f1-score, etc.
    confusion_matrix,                                                 # - Matriz de confusión para ver aciertos y errores por clase.
    accuracy_score,                                                   # - Porcentaje de aciertos totales.
    roc_auc_score,                                                    # - Área bajo la curva ROC.
    roc_curve,                                                        # - Puntos para trazar la curva ROC.
    auc                                                               # - Cálculo del área bajo cualquier curva dada (como la ROC).
)
from sklearn.preprocessing import StandardScaler                      # Importa el escalador estándar para normalizar los datos.
warnings.filterwarnings("ignore")                                     # Desactiva las advertencias para evitar que se muestren en la salida del programa.

# =============================================================================
# CARGA Y PREPARACIÓN DE DATOS
# =============================================================================

# 1. Cargar el dataset
df = pd.read_csv('heart_attack_desease.csv')                         # Carga el archivo CSV con los datos sobre enfermedades cardíacas en un DataFrame.

# 2. Revisar valores nulos
print("Valores nulos por columna:\n", df.isnull().sum())            # Muestra cuántos valores nulos hay por columna para detectar datos faltantes.

# 3. Separar características (X) y etiqueta (y)
X = df.drop('target', axis=1)                                        # Separa las características del modelo, eliminando la columna 'target'.
y = df['target']                                                     # Asigna la columna 'target' como variable objetivo (etiqueta a predecir).

# 4. Escalar los datos
scaler = StandardScaler()                                            # Crea un objeto escalador que normaliza los datos (media 0, desviación estándar 1).
X_scaled = scaler.fit_transform(X)                                   # Ajusta el escalador a los datos y transforma las características escalándolas.

# 5. Dividir en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(                 # Divide los datos en conjunto de entrenamiento y prueba:
    X_scaled, y, test_size=0.2, random_state=42, stratify=y)         # - 80% para entrenamiento, 20% para prueba, con distribución balanceada según 'y'.

# =============================================================================
# FUNCIONES AUXILIARES
# =============================================================================

# Obtiene el mejor modelo que no supere la precisión máxima especificada
def obtener_mejor_modelo_restringido(modelo_grid, nombre, X_train, y_train, max_precision=0.96):  # Define una función que selecciona el mejor modelo limitado por una precisión máxima.
    results = modelo_grid.cv_results_                                                              # Obtiene todos los resultados de validación cruzada del GridSearchCV.

    if nombre in ["KNN", "Random Forest"]:                                                         # Solo aplica esta lógica personalizada si el modelo es KNN o Random Forest.
        # Filtrar configuraciones que no superen la precisión máxima
        valid_indices = [i for i, score in enumerate(results['mean_test_score']) if score <= max_precision]  # Guarda los índices de las configuraciones que cumplen con la restricción de precisión.

        if len(valid_indices) == 0:                                                                # Si no hay configuraciones válidas (todas superan el límite de precisión)...
            print(f"Advertencia: Ninguna configuración de {nombre} está por debajo de {max_precision*100}%")  # ...muestra advertencia...
            print(f"Usando la configuración con menor precisión disponible")                       # ...e informa que se usará la menos precisa.
            best_idx = np.argmin(results['mean_test_score'])                                       # Toma el índice del modelo con menor precisión (por si acaso hay sobreajuste extremo).
        else:
            # Ordenar por precisión y tomar el mejor
            valid_scores = [(results['mean_test_score'][i], i) for i in valid_indices]             # Crea lista de tuplas (precisión, índice) para configuraciones válidas.
            valid_scores.sort(reverse=True)                                                        # Ordena las tuplas de mayor a menor precisión.
            best_idx = valid_scores[0][1]                                                           # Toma el índice de la mejor configuración dentro del límite permitido.

        # Obtener los parámetros del mejor modelo válido
        best_params = results['params'][best_idx]                                                  # Extrae los parámetros del mejor modelo.
        best_cv_score = results['mean_test_score'][best_idx]                                       # Extrae la precisión de validación cruzada del mejor modelo.

        # Crear y entrenar el modelo con esos parámetros
        if nombre == "KNN":
            best_model = KNeighborsClassifier(**best_params)                                       # Si es KNN, crea un modelo con los mejores parámetros encontrados.
        elif nombre == "Random Forest":
            best_model = RandomForestClassifier(**best_params, random_state=42)                    # Si es Random Forest, crea el modelo con los mejores parámetros y semilla fija.

        best_model.fit(X_train, y_train)                                                            # Entrena el modelo seleccionado con los datos de entrenamiento.

        print(f"Mejor {nombre} (restringido): {best_params}")                                      # Muestra los parámetros del mejor modelo restringido.
        print(f"Precisión CV: {best_cv_score:.4f}")                                                # Muestra la precisión obtenida en validación cruzada.

        return best_model, best_params, best_cv_score                                              # Retorna el modelo entrenado, sus parámetros y su precisión de validación.

    else:
        # Para regresión logística, usar el comportamiento original
        return modelo_grid.best_estimator_, modelo_grid.best_params_, modelo_grid.best_score_      # Si no es KNN ni Random Forest, retorna el mejor modelo como lo entrega GridSearchCV.

# Evalúa y muestra los mejores n modelos con restricción de precisión
def top_n_modelos_restringido(modelo_grid, nombre, X_train, y_train, X_test, y_test, n=4, max_precision=0.96):  # Define una función para mostrar los top n modelos con precisión ≤ max_precision.
    results = modelo_grid.cv_results_                                              # Extrae todos los resultados de validación cruzada del GridSearchCV.

    # Filtrar configuraciones que no superen la precisión máxima
    if nombre in ["KNN", "Random Forest"]:                                         # Solo aplica la lógica de restricción si el modelo es KNN o Random Forest.
        valid_indices = [i for i, score in enumerate(results['mean_test_score']) if score <= max_precision]  # Filtra índices con precisión menor o igual al límite.

        if len(valid_indices) < n:                                                 # Si hay menos modelos válidos que n...
            print(f"Advertencia: Solo {len(valid_indices)} configuraciones de {nombre} no superan {max_precision*100}% de precisión")  # ...avisa al usuario.
            valid_indices = valid_indices if valid_indices else list(range(len(results['mean_test_score'])))  # Si no hay ninguna válida, se usan todas como alternativa.

        # Ordenar por precisión y tomar los mejores n
        valid_scores = [(results['mean_test_score'][i], i) for i in valid_indices]  # Crea una lista de tuplas (precisión, índice) para los modelos válidos.
        valid_scores.sort(reverse=True)                                             # Ordena de mayor a menor precisión.
        sorted_indices = [idx for _, idx in valid_scores[:n]]                       # Toma los índices de los mejores n modelos dentro del límite.

    else:
        # Para regresión logística, usar el comportamiento original
        sorted_indices = np.argsort(results['rank_test_score'])[:n]                # Para otros modelos (como regresión logística), toma los top n mejores según ranking de GridSearch.

    print(f"\n\n Top {min(len(sorted_indices), n)} resultados para {nombre} (máx. {max_precision*100}% precisión):")  # Imprime cuántos modelos se mostrarán y su restricción.

    for rank, idx in enumerate(sorted_indices, start=1):                           # Itera sobre los índices ordenados, con numeración desde 1.
        params = results['params'][idx]                                            # Obtiene los hiperparámetros de la configuración actual.
        cv_precision = results['mean_test_score'][idx]                             # Obtiene la precisión media en validación cruzada.

        # Según el nombre del modelo, instancia el clasificador con esos hiperparámetros
        if nombre == "Regresión Logística":
            modelo = LogisticRegression(**params, random_state=42)                # Crea modelo de regresión logística con los parámetros y semilla fija.
        elif nombre == "KNN":
            modelo = KNeighborsClassifier(**params)                               # Crea modelo KNN con los parámetros seleccionados.
        elif nombre == "Random Forest":
            modelo = RandomForestClassifier(**params, random_state=42)            # Crea modelo Random Forest con los parámetros y semilla fija.
        else:
            continue                                                               # Si el nombre no coincide con ninguno conocido, omite la iteración.

        modelo.fit(X_train, y_train)                                               # Entrena el modelo con el conjunto de entrenamiento.

        y_pred = modelo.predict(X_test)                                            # Realiza predicciones de clase sobre el conjunto de prueba.
        y_proba = modelo.predict_proba(X_test)[:, 1]                               # Obtiene las probabilidades para la clase positiva (para calcular AUC).

        print(f"\n--- {nombre} (Top #{rank}) ---")                                 # Imprime el encabezado del modelo actual en el ranking.
        print("Configuración:", params)                                            # Muestra los hiperparámetros usados.
        print(f"Precisión CV: {cv_precision:.4f}")                                 # Muestra la precisión en validación cruzada.
        print("Matriz de confusión:\n", confusion_matrix(y_test, y_pred))         # Muestra la matriz de confusión del modelo sobre el test set.
        print("Reporte de clasificación:\n", classification_report(y_test, y_pred))  # Imprime métricas detalladas: precisión, recall, F1-score.
        print("Precisión test:", accuracy_score(y_test, y_pred))                   # Imprime la precisión del modelo en el conjunto de prueba.
        print("AUC:", roc_auc_score(y_test, y_proba))                              # Imprime el área bajo la curva ROC, que mide rendimiento general del clasificador.

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

    valores = []                                                            # Crea una lista vacía donde se guardarán los valores ingresados por el usuario.

    for var, pregunta in preguntas.items():                                 # Itera sobre cada par (clave, pregunta) del diccionario 'preguntas'.
        valor = float(input(pregunta))                                      # Muestra la pregunta por consola y convierte la respuesta a tipo float.
        valores.append(valor)                                               # Agrega el valor ingresado a la lista de valores.

        entrada_np = np.array(valores).reshape(1, -1)                           # Convierte la lista a un array NumPy y le da forma de una fila con muchas columnas.

        scaler = joblib.load('scaler.pkl')                                      # Carga el objeto de escalado previamente guardado (StandardScaler, por ejemplo).
        modelo = joblib.load('rf_model.pkl')                                    # Carga el modelo de Random Forest entrenado previamente.

        entrada_scaled = scaler.transform(entrada_np)                           # Escala la entrada del usuario con el mismo escalador usado en el entrenamiento.
        proba = modelo.predict_proba(entrada_scaled)[0][1]                      # Calcula la probabilidad de clase positiva (riesgo de enfermedad cardíaca).

        if proba > 0.6:                                                         # Si la probabilidad es mayor a 60%...
            print("\n🔴 Riesgo alto de enfermedad cardíaca.")                    # ...se muestra un mensaje de advertencia de riesgo alto.
        else:
            print("\n🟢 Bajo riesgo de enfermedad cardíaca.")                    # Si no, se indica que el riesgo es bajo.

# =============================================================================
# ENTRENAMIENTO Y BÚSQUEDA DE HIPERPARÁMETROS
# =============================================================================

# 6. Buscar mejores hiperparámetros y entrenar modelos

# Logistic Regression
log_params = {'C': np.logspace(-3, 3, 10)}                                # Define un diccionario de hiperparámetros para la regresión logística (valores de C en escala logarítmica de 0.001 a 1000).
log_model = GridSearchCV(LogisticRegression(random_state=42), log_params, cv=5)  # Crea un GridSearchCV con validación cruzada de 5 pliegues para buscar el mejor valor de C.
log_model.fit(X_train, y_train)                                          # Ajusta el modelo con los datos de entrenamiento.
log_best = log_model.best_estimator_                                     # Guarda el mejor modelo encontrado en la búsqueda de hiperparámetros.
log_pred = log_best.predict(X_test)                                      # Genera las predicciones del mejor modelo sobre el conjunto de prueba.
log_proba = log_best.predict_proba(X_test)[:, 1]                         # Obtiene las probabilidades de pertenecer a la clase positiva (enfermedad cardíaca).

# KNN con restricción de precisión máxima del 96%
knn_params = {'n_neighbors': list(range(1, 10))}                         # Define los valores a probar para el hiperparámetro n_neighbors (de 1 a 9).
knn_model = GridSearchCV(KNeighborsClassifier(), knn_params, cv=30)     # Crea una búsqueda de hiperparámetros para KNN con validación cruzada de 30 pliegues.
knn_model.fit(X_train, y_train)                                          # Entrena el modelo usando los datos de entrenamiento.

# Obtener el mejor modelo KNN que no supere 96%
knn_best, knn_best_params, knn_best_cv_score = obtener_mejor_modelo_restringido(  # Usa una función personalizada para obtener el mejor KNN con precisión CV ≤ 96%.
    knn_model, "KNN", X_train, y_train, max_precision=0.96)
knn_pred = knn_best.predict(X_test)                                     # Predice las clases en el conjunto de prueba usando el mejor KNN.
knn_proba = knn_best.predict_proba(X_test)[:, 1]                         # Obtiene las probabilidades de la clase positiva.

# Random Forest con restricción de precisión máxima del 96%
rf_params = {'max_depth': list(range(1, 10))}                            # Define los valores a probar para la profundidad máxima del árbol (1 a 9).
rf_model = GridSearchCV(RandomForestClassifier(random_state=42), rf_params, cv=30)  # Crea una búsqueda de hiperparámetros para Random Forest con validación cruzada de 30 pliegues.
rf_model.fit(X_train, y_train)                                          # Entrena el modelo Random Forest con los datos de entrenamiento.

# Obtener el mejor modelo Random Forest que no supere 96%
rf_best, rf_best_params, rf_best_cv_score = obtener_mejor_modelo_restringido(  # Usa la función personalizada para seleccionar el mejor modelo bajo la restricción de precisión.
    rf_model, "Random Forest", X_train, y_train, max_precision=0.96)
rf_pred = rf_best.predict(X_test)                                       # Realiza predicciones sobre el conjunto de prueba con el mejor modelo.
rf_proba = rf_best.predict_proba(X_test)[:, 1]                           # Obtiene las probabilidades de clase positiva.

# =============================================================================
# EVALUACIÓN DE MODELOS
# =============================================================================

# 7. Evaluar múltiples configuraciones con restricción de precisión
top_n_modelos_restringido(log_model, "Regresión Logística", X_train, y_train, X_test, y_test, n=4)       # Evalúa y muestra las 4 mejores configuraciones de regresión logística (sin restricción de precisión).
top_n_modelos_restringido(knn_model, "KNN", X_train, y_train, X_test, y_test, n=4, max_precision=0.96)    # Evalúa y muestra las 4 mejores configuraciones de KNN con precisión ≤ 96%.
top_n_modelos_restringido(rf_model, "Random Forest", X_train, y_train, X_test, y_test, n=4, max_precision=0.96)  # Evalúa y muestra las 4 mejores configuraciones de Random Forest con precisión ≤ 96%.

# Preparar probabilidades para clase 0
log_proba_0 = log_best.predict_proba(X_test)[:, 0]           # Obtiene las probabilidades de la clase 0 (sin enfermedad) para el modelo de regresión logística.
knn_proba_0 = knn_best.predict_proba(X_test)[:, 0]           # Obtiene las probabilidades de la clase 0 para el mejor modelo KNN.
rf_proba_0 = rf_best.predict_proba(X_test)[:, 0]             # Obtiene las probabilidades de la clase 0 para el mejor modelo Random Forest.

# Invertir las etiquetas: ahora la clase "0" es la positiva
y_test_invertido = 1 - y_test                                # Invierte las etiquetas reales del conjunto de prueba: ahora "0" (sin enfermedad) se considera la clase positiva.

# =============================================================================
# VISUALIZACIONES
# =============================================================================

# 8. Graficar curvas ROC para la clase 0
plt.figure(figsize=(8, 6))                                                             # Crea una nueva figura de tamaño 8x6 para el gráfico.

for nombre, y_proba in zip(["Logistic Regression", "KNN", "Random Forest"],           # Itera sobre los modelos y sus probabilidades para la clase 0...
                          [log_proba_0, knn_proba_0, rf_proba_0]):
    fpr, tpr, _ = roc_curve(y_test_invertido, y_proba)                                 # Calcula tasa de falsos positivos (FPR) y verdaderos positivos (TPR) para la clase 0.
    plt.plot(fpr, tpr,                                                                 # Traza la curva ROC para cada modelo.
             label=f'{nombre} (AUC: {int(roc_auc_score(y_test_invertido, y_proba) * 100) / 100:.2f})')  # Añade la AUC truncada a 2 decimales como etiqueta.

plt.plot([0, 1], [0, 1], 'k--')                                                        # Traza una línea diagonal punteada como referencia (modelo aleatorio).
plt.xlabel('FPR (Clase 0)')                                                            # Etiqueta eje X: tasa de falsos positivos.
plt.ylabel('TPR (Clase 0)')                                                            # Etiqueta eje Y: tasa de verdaderos positivos.
plt.title('Curvas ROC [Clase 0]')                                                      # Título del gráfico.
plt.legend()                                                                           # Muestra la leyenda con los nombres de los modelos.
plt.grid(True)                                                                         # Activa la cuadrícula para facilitar la lectura.
plt.tight_layout()                                                                     # Ajusta automáticamente los márgenes del gráfico.
plt.show()                                                                             # Muestra el gráfico en pantalla.

# 8.1. Graficar curvas ROC para la clase 1
plt.figure(figsize=(8, 6))                                                             # Crea una nueva figura para graficar la clase 1.

for nombre, y_proba in zip(["Logistic Regression", "KNN", "Random Forest"], 
                          [log_proba, knn_proba, rf_proba]):                           # Itera sobre los modelos y sus probabilidades para la clase 1.
    fpr, tpr, _ = roc_curve(y_test, y_proba)                                           # Calcula FPR y TPR usando las etiquetas originales (clase 1 como positiva).
    plt.plot(fpr, tpr,                                                                 # Traza la curva ROC...
             label=f'{nombre} (AUC: {int(roc_auc_score(y_test, y_proba) * 100) / 100:.2f})')  # ...con AUC truncada en la etiqueta.

plt.plot([0, 1], [0, 1], 'k--')                                                        # Línea de referencia diagonal (modelo aleatorio).
plt.xlabel('FPR (Clase 1)')                                                            # Etiqueta del eje X.
plt.ylabel('TPR (Clase 1)')                                                            # Etiqueta del eje Y.
plt.title('Curvas ROC [Clase 1]')                                                      # Título del gráfico.
plt.legend()                                                                           # Muestra la leyenda de los modelos.
plt.grid(True)                                                                         # Activa cuadrícula.
plt.tight_layout()                                                                     # Ajusta márgenes automáticamente.
plt.show()                                                                             # Muestra el gráfico.

# 8.2 Curvas ROC para KNN (mejores modelos)
fpr_knn, tpr_knn, _ = roc_curve(y_test, knn_proba)                                     # Calcula FPR y TPR para la clase 1 usando el mejor modelo KNN.
roc_auc_knn = auc(fpr_knn, tpr_knn)                                                    # Calcula el área bajo la curva (AUC) para clase 1.

fpr_knn_0, tpr_knn_0, _ = roc_curve(y_test_invertido, knn_proba_0)                     # Calcula FPR y TPR para la clase 0.
roc_auc_knn_0 = auc(fpr_knn_0, tpr_knn_0)                                               # Calcula AUC para la clase 0.

plt.figure(figsize=(8, 6))                                                             # Crea una nueva figura para graficar ambas curvas del KNN.
plt.plot(fpr_knn, tpr_knn, label=f'Clase 1 (AUC = {roc_auc_knn:.2f})')                 # Traza curva ROC para clase 1 y muestra su AUC.
plt.plot(fpr_knn_0, tpr_knn_0, label=f'Clase 0 (AUC = {roc_auc_knn_0:.2f})')           # Traza curva ROC para clase 0 con su AUC.
plt.plot([0, 1], [0, 1], 'k--')                                                        # Línea diagonal de referencia.
plt.xlabel('FPR')                                                                      # Etiqueta eje X (general).
plt.ylabel('TPR')                                                                      # Etiqueta eje Y.
plt.title(f'Curvas ROC del mejor modelo KNN (Restricción 96%) - {knn_best_params}')   # Título con hiperparámetros del mejor modelo KNN.
plt.legend(loc='lower right')                                                          # Muestra leyenda abajo a la derecha.
plt.grid(True)                                                                         # Cuadrícula activada.
plt.tight_layout()                                                                     # Ajuste automático de márgenes.
plt.show()                                                                             # Muestra el gráfico.

# 8.3 Curvas ROC para Random Forest (mejores modelos)
fpr_rf, tpr_rf, _ = roc_curve(y_test, rf_proba)                                # Calcula FPR y TPR para la clase 1 (enfermedad) con el modelo Random Forest.
roc_auc_rf = auc(fpr_rf, tpr_rf)                                               # Calcula el área bajo la curva (AUC) para la clase 1.

fpr_rf_0, tpr_rf_0, _ = roc_curve(y_test_invertido, rf_proba_0)                # Calcula FPR y TPR para la clase 0 (sanos) usando etiquetas invertidas.
roc_auc_rf_0 = auc(fpr_rf_0, tpr_rf_0)                                         # Calcula AUC para la clase 0.

plt.figure(figsize=(8, 6))                                                     # Crea una nueva figura para el gráfico.
plt.plot(fpr_rf, tpr_rf, label=f'Clase 1 (AUC = {roc_auc_rf:.2f})')            # Dibuja curva ROC para clase 1.
plt.plot(fpr_rf_0, tpr_rf_0, label=f'Clase 0 (AUC = {roc_auc_rf_0:.2f})')      # Dibuja curva ROC para clase 0.
plt.plot([0, 1], [0, 1], 'k--')                                                # Línea diagonal como referencia (modelo aleatorio).
plt.xlabel('FPR')                                                              # Etiqueta del eje X.
plt.ylabel('TPR')                                                              # Etiqueta del eje Y.
plt.title(f'Curvas ROC del mejor modelo Random Forest (Restricción 96%) - {rf_best_params}')  # Título del gráfico con los parámetros óptimos.
plt.legend(loc='lower right')                                                  # Coloca leyenda en la esquina inferior derecha.
plt.grid(True)                                                                 # Activa la cuadrícula.
plt.tight_layout()                                                             # Ajusta márgenes automáticamente.
plt.show()                                                                     # Muestra el gráfico.

# 8.4 Curvas ROC para Regresión Logística
fpr_log, tpr_log, _ = roc_curve(y_test, log_proba)                             # Calcula FPR y TPR para clase 1 con regresión logística.
roc_auc_log = auc(fpr_log, tpr_log)                                            # Calcula AUC para clase 1.

fpr_log_0, tpr_log_0, _ = roc_curve(y_test_invertido, log_proba_0)            # Calcula FPR y TPR para clase 0 con etiquetas invertidas.
roc_auc_log_0 = auc(fpr_log_0, tpr_log_0)                                     # Calcula AUC para clase 0.

plt.figure(figsize=(8, 6))                                                     # Nueva figura.
plt.plot(fpr_log, tpr_log, label=f'Clase 1 (AUC = {roc_auc_log:.2f})')        # Dibuja curva ROC para clase 1.
plt.plot(fpr_log_0, tpr_log_0, label=f'Clase 0 (AUC = {roc_auc_log_0:.2f})')  # Dibuja curva ROC para clase 0.
plt.plot([0, 1], [0, 1], 'k--')                                                # Línea aleatoria de referencia.
plt.xlabel('FPR')                                                              # Etiqueta eje X.
plt.ylabel('TPR')                                                              # Etiqueta eje Y.
plt.title('Curvas ROC del mejor modelo Regresión Logística')                  # Título del gráfico.
plt.legend(loc='lower right')                                                  # Leyenda abajo a la derecha.
plt.grid(True)                                                                 # Activa cuadrícula.
plt.tight_layout()                                                             # Ajusta el diseño.
plt.show()                                                                     # Muestra el gráfico.

# 9. Matrices de confusión heatmap
modelos = [("Logistic Regression", log_pred), ("KNN (≤96%)", knn_pred), ("Random Forest (≤96%)", rf_pred)]  # Lista de tuplas con nombre del modelo y sus predicciones.

for nombre, pred in modelos:                                                   # Itera sobre cada modelo y sus predicciones.
    plt.figure()                                                               # Crea una nueva figura para cada matriz.
    sns.heatmap(confusion_matrix(y_test, pred),                               # Dibuja la matriz de confusión como un heatmap.
                annot=True, fmt='d', cmap='Blues')                            # Muestra los números en cada celda, con color azul.
    plt.title(f'Matriz de Confusión - {nombre}')                              # Título del gráfico.
    plt.xlabel('Predicción')                                                  # Etiqueta del eje X.
    plt.ylabel('Real')                                                        # Etiqueta del eje Y.
    plt.tight_layout()                                                        # Ajuste automático del diseño.
    plt.show()                                                                # Muestra el gráfico.


# 10. Curvas de precisión vs hiperparámetros

# KNN - Mostrar solo configuraciones que no superen 96%
plt.figure()                                                                # Crea una nueva figura para graficar.
knn_scores = knn_model.cv_results_['mean_test_score']                      # Obtiene las precisiones promedio de cada configuración KNN.
knn_params_values = knn_params['n_neighbors']                              # Obtiene la lista de valores de K probados.

# Filtrar valores que no superen 96%
knn_filtered_scores = []                                                   # Lista para almacenar precisiones ≤ 96%.
knn_filtered_params = []                                                   # Lista para almacenar valores de K correspondientes.
for i, (param, score) in enumerate(zip(knn_params_values, knn_scores)):    # Itera sobre pares (K, precisión).
    if score <= 0.96:                                                      # Si la precisión es menor o igual a 96%...
        knn_filtered_params.append(param)                                 # Guarda el valor de K.
        knn_filtered_scores.append(score)                                 # Guarda la precisión.

plt.plot(knn_filtered_params, knn_filtered_scores, marker='o', label='≤96% precisión')  # Grafica solo las precisiones bajo 96% con círculos.
plt.plot(knn_params_values, knn_scores, marker='x', alpha=0.5, label='Todos los valores')  # Grafica todas las precisiones con cruces y transparencia.

# Marcar el mejor modelo seleccionado
if knn_best_params['n_neighbors'] in knn_filtered_params:                  # Si el mejor KNN seleccionado está en los filtrados...
    best_score = knn_scores[knn_params_values.index(knn_best_params['n_neighbors'])]  # Obtiene su precisión.
    plt.plot(knn_best_params['n_neighbors'], best_score, marker='*', markersize=15,  # Lo marca en la gráfica con una estrella roja grande.
             color='red', label=f'Mejor seleccionado (k={knn_best_params["n_neighbors"]})')

plt.title("Precisión KNN vs K (con restricción 96%)")                      # Título del gráfico.
plt.xlabel("Valores de K")                                                 # Etiqueta eje X.
plt.ylabel("Precisión")                                                    # Etiqueta eje Y.
plt.axhline(y=0.96, color='r', linestyle='--', label='Límite 96%')          # Dibuja línea horizontal roja en y=0.96 como referencia.
plt.legend()                                                              # Muestra leyenda.
plt.grid(True)                                                            # Activa cuadrícula.
plt.tight_layout()                                                        # Ajusta márgenes.
plt.show()                                                              # Muestra el gráfico.


# Random Forest - Mostrar solo configuraciones que no superen 96%
plt.figure()                                                               # Nueva figura para Random Forest.
rf_scores = rf_model.cv_results_['mean_test_score']                       # Obtiene precisiones promedio de cada configuración de Random Forest.
rf_params_values = rf_params['max_depth']                                # Lista de valores de profundidad máxima probados.

# Filtrar valores que no superen 96%
rf_filtered_scores = []                                                   # Lista para precisiones ≤ 96%.
rf_filtered_params = []                                                   # Lista para profundidades correspondientes.
for i, (param, score) in enumerate(zip(rf_params_values, rf_scores)):     # Itera por pares (profundidad, precisión).
    if score <= 0.96:                                                     # Si precisión ≤ 96%...
        rf_filtered_params.append(param)                                 # Guarda profundidad.
        rf_filtered_scores.append(score)                                 # Guarda precisión.

plt.plot(rf_filtered_params, rf_filtered_scores, marker='o', color='green', label='≤96% precisión')  # Grafica puntos verdes para precisiones bajo 96%.
plt.plot(rf_params_values, rf_scores, marker='x', color='green', alpha=0.5, label='Todos los valores')  # Grafica todas precisiones con cruces verdes y transparencia.

# Marcar el mejor modelo seleccionado
if rf_best_params['max_depth'] in rf_filtered_params:                    # Si el mejor parámetro está entre los filtrados...
    best_score = rf_scores[rf_params_values.index(rf_best_params['max_depth'])]  # Obtiene la precisión del mejor modelo.
    plt.plot(rf_best_params['max_depth'], best_score, marker='*', markersize=15,    # Marca el punto con estrella roja grande.
             color='red', label=f'Mejor seleccionado (depth={rf_best_params["max_depth"]})')

plt.title("Precisión RF vs Max Depth (con restricción 96%)")              # Título del gráfico.
plt.xlabel("Profundidad máxima")                                         # Etiqueta eje X.
plt.ylabel("Precisión")                                                  # Etiqueta eje Y.
plt.axhline(y=0.96, color='r', linestyle='--', label='Límite 96%')        # Línea horizontal roja en 0.96.
plt.legend()                                                            # Muestra leyenda.
plt.grid(True)                                                          # Activa cuadrícula.
plt.tight_layout()                                                    # Ajusta márgenes.
plt.show()                                                          # Muestra el gráfico.

# Regresión Logística (sin restricción)
accuracies_test = []                                                        # Inicializa una lista vacía para guardar las precisiones en test.

for c in log_params['C']:                                                   # Itera sobre cada valor de C definido en log_params.
    model = LogisticRegression(C=c, max_iter=1000)                          # Crea un modelo de regresión logística con el parámetro C actual y máximo de 1000 iteraciones.
    model.fit(X_train, y_train)                                             # Entrena el modelo con los datos de entrenamiento.
    acc = accuracy_score(y_test, model.predict(X_test))                     # Calcula la precisión del modelo usando las predicciones sobre el conjunto de prueba.
    accuracies_test.append(acc)                                             # Añade la precisión calculada a la lista accuracies_test.

plt.figure()                                                               # Crea una nueva figura para el gráfico.
plt.plot(log_params['C'], accuracies_test, marker='o', color='red')        # Dibuja una curva de precisión vs valor de C con marcadores rojos.
plt.xscale('log')                                                          # Establece escala logarítmica en el eje X (para los valores de C).
plt.title("Precisión Regresión Logística vs C")                           # Añade título al gráfico.
plt.xlabel("Valores de C (log)")                                           # Etiqueta el eje X.
plt.ylabel("Precisión")                                                    # Etiqueta el eje Y.
plt.grid(True)                                                             # Activa la cuadrícula para mejor visualización.
plt.tight_layout()                                                         # Ajusta el diseño para que no se corten elementos.
plt.show()                                                                 # Muestra el gráfico en pantalla.

# =============================================================================
# GUARDADO DE MODELOS Y ANÁLISIS FINAL
# =============================================================================

# 11. Guardar mejor modelo y scaler
joblib.dump(rf_best, 'rf_model.pkl')                      # Guarda el mejor modelo Random Forest entrenado en un archivo llamado 'rf_model.pkl'.
joblib.dump(scaler, 'scaler.pkl')                         # Guarda el objeto scaler (normalizador) en un archivo llamado 'scaler.pkl'.

print(f"\n✅ Modelo Random Forest guardado con restricción 96%:")  # Imprime mensaje confirmando que el modelo se guardó correctamente.
print(f"Parámetros: {rf_best_params}")                               # Muestra los parámetros del mejor modelo Random Forest guardado.
print(f"Precisión CV: {rf_best_cv_score:.4f}")                       # Muestra la precisión promedio en validación cruzada con 4 decimales.
print(f"Precisión Test: {accuracy_score(y_test, rf_pred):.4f}")     # Muestra la precisión obtenida en el conjunto de prueba con 4 decimales.

# 12. Predicción interactiva (descomenta para usar)
# entrada_usuario()

# =============================================================================
# ANÁLISIS DE IMPORTANCIA DE CARACTERÍSTICAS
# =============================================================================

print("\n" + "="*60)                                              # Imprime una línea de 60 signos "=" para separar visualmente.
print("📊 ANÁLISIS DE IMPORTANCIA DE CARACTERÍSTICAS")           # Imprime título del análisis con emoji.
print("="*60)                                                    # Imprime otra línea de separación.

print("\n7️⃣ CORRELACIÓN ABSOLUTA CON LA VARIABLE OBJETIVO:")     # Imprime subtítulo para esta sección del análisis.
print("-" * 50)                                                  # Imprime una línea de 50 guiones para separar visualmente.

# Calcular correlación absoluta con la variable objetivo
correlations = X.corrwith(y).abs().sort_values(ascending=False)  # Calcula la correlación entre cada característica (columna de X) y la variable objetivo y, toma el valor absoluto y ordena de mayor a menor.
correlation_df = pd.DataFrame({                                  # Crea un DataFrame para organizar los resultados.
    'Característica': correlations.index,                       # Columna con nombres de las características.
    'Correlación_Abs': correlations.values,                     # Columna con los valores absolutos de correlación.
    'Porcentaje_Corr': (correlations.values / correlations.sum()) * 100  # Calcula el porcentaje que representa cada correlación respecto al total de correlaciones absolutas.
})

# Mostrar correlaciones con target
print("Correlación absoluta con variable objetivo:")            # Imprime encabezado antes de listar resultados.
for i, row in correlation_df.iterrows():                        # Itera sobre cada fila del DataFrame de correlaciones.
    print(f"{i+1:2d}. {row['Característica']:12s} - {row['Correlación_Abs']:.3f} ({row['Porcentaje_Corr']:5.2f}%)")  # Imprime índice, nombre característica, correlación absoluta con 3 decimales y porcentaje con 2 decimales.

print("\n" + "="*60)                                            # Imprime línea de separación.
print("✅ ANÁLISIS COMPLETADO")                                  # Mensaje indicando que el análisis terminó.
print("="*60)                                                   # Otra línea de separación.