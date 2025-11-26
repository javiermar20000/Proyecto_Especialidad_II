# Predicción de Enfermedades Cardíacas

Modelo de clasificación para estimar riesgo de enfermedad cardíaca a partir de variables clínicas (edad, presión, colesterol, resultados ECG, etc.). El proyecto entrena y compara varios algoritmos, guarda el mejor modelo junto al `StandardScaler` y permite hacer predicciones manuales desde consola.

## Contenido del repositorio
- `heart_attack_desease.csv`: dataset tabular con 303 registros y 14 columnas (13 características + `target`).
- `prediccion_enfermedades_cardiacas.py`: script principal de experimentación con búsqueda de hiperparámetros y visualizaciones.
- `prediccion_enfermedades_cardiacas_CV.py`: variante con validación cruzada más estricta (cv=30) y límite de precisión (≤96%) para evitar sobreajuste.
- `rf_model.pkl`, `randomforest_model.pkl`: modelos Random Forest entrenados (el primero es el más reciente según los scripts).
- `scaler.pkl`: `StandardScaler` ajustado sobre el dataset.
- `enlace_video.txt`: enlace al video explicativo (YouTube).

## Dataset
Columnas disponibles en `heart_attack_desease.csv`:
- `age`: edad en años.
- `sex`: 1 hombre, 0 mujer.
- `cp`: tipo de dolor en el pecho (0–3).
- `trestbps`: presión arterial en reposo (mm Hg).
- `chol`: colesterol sérico (mg/dl).
- `fbs`: azúcar en sangre en ayunas > 120 mg/dl (1 sí, 0 no).
- `restecg`: resultados ECG en reposo (0–2).
- `thalach`: frecuencia cardíaca máxima alcanzada.
- `exang`: angina inducida por ejercicio (1 sí, 0 no).
- `oldpeak`: depresión ST inducida por ejercicio.
- `slope`: pendiente del segmento ST (0–2).
- `ca`: número de vasos principales coloreados (0–3).
- `thal`: resultado del test Thal (1 normal, 2 fijo, 3 reversible).
- `target`: etiqueta binaria (1 = riesgo/enfermedad, 0 = sin enfermedad).

## Flujo de trabajo
1) Cargar y explorar datos: detección de nulos y separación en `X`/`y`.
2) Escalado: `StandardScaler` para normalizar características.
3) Partición: `train_test_split` estratificado (80/20).
4) Búsqueda de hiperparámetros:
   - Regresión Logística (`C` en escala log).
   - KNN (`n_neighbors`).
   - Random Forest (`max_depth`).
5) Evaluación: matrices de confusión, `classification_report`, accuracy, AUC y curvas ROC por clase.
6) Análisis adicional:
   - Curvas precisión vs. hiperparámetros.
   - Correlación absoluta de cada variable con `target` (solo en `*_CV.py`).
7) Persistencia: guarda el mejor Random Forest (`rf_model.pkl`) y el `scaler.pkl`.
8) Predicción interactiva: formulario en consola para evaluar un paciente (usa el modelo y scaler guardados).

## Diferencias entre scripts
- `prediccion_enfermedades_cardiacas.py`
  - Validación cruzada estándar (cv=5).
  - Evalúa top‑N configuraciones de cada modelo y grafica ROC/matrices de confusión.
- `prediccion_enfermedades_cardiacas_CV.py`
  - Validación cruzada más exigente (cv=30).
  - Restringe modelos KNN y RF a precisión ≤96% para evitar sobreajuste.
  - Añade análisis de correlación con la variable objetivo.

## Requisitos
- Python 3.9+  
- Librerías: `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, `joblib`.

Instalación rápida en un entorno virtual:
```bash
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -U pip
pip install pandas numpy scikit-learn matplotlib seaborn joblib
```

## Cómo ejecutar los experimentos
Ejecuta cualquiera de los scripts desde la raíz del proyecto:
```bash
python3 prediccion_enfermedades_cardiacas.py
# o
python3 prediccion_enfermedades_cardiacas_CV.py
```

Al finalizar, se generan/actualizan:
- `rf_model.pkl`: mejor modelo Random Forest encontrado.
- `scaler.pkl`: escalador usado en entrenamiento.

## Uso de predicción manual
1) Asegúrate de tener `rf_model.pkl` y `scaler.pkl` generados.
2) Descomenta la función `entrada_usuario()` al final del script que prefieras.
3) Ejecuta el script y responde las preguntas en consola.  
   - Si la probabilidad > 0.6 → riesgo alto (`target=1`).
   - De lo contrario → riesgo bajo (`target=0`).

## Resultados y métricas
- Ambos scripts imprimen:
  - Matrices de confusión.
  - `classification_report` por clase.
  - Accuracy y AUC en test.
  - Curvas ROC para clases 0 y 1.
- La versión `*_CV` además muestra:
  - Precisión de validación cruzada por hiperparámetro.
  - Correlación absoluta de cada variable con el `target`.

## Video explicativo
En `enlace_video.txt` encontrarás el enlace a la presentación del proyecto:
https://youtu.be/gzEI4akXink

## Estructura de archivos
```
Proyecto_Especialidad_II/
├── heart_attack_desease.csv
├── prediccion_enfermedades_cardiacas.py
├── prediccion_enfermedades_cardiacas_CV.py
├── rf_model.pkl
├── randomforest_model.pkl
├── scaler.pkl
├── enlace_video.txt
└── README.md
```

¡Listo! Con esto puedes reproducir los experimentos, ajustar hiperparámetros o usar el modelo entrenado para nuevas predicciones.
