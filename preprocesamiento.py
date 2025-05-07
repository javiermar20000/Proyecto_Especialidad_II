import pandas as pd  # Importa pandas para manipulación de datos en forma de DataFrames
import numpy as np  # Importa numpy para operaciones matemáticas y manejo de arrays
import seaborn as sns  # Importa seaborn para visualización estadística más atractiva
import matplotlib.pyplot as plt  # Importa matplotlib para generar gráficos
from scipy import stats  # Importa funciones estadísticas de scipy
from sklearn.decomposition import PCA  # Importa PCA para análisis de componentes principales
from sklearn.preprocessing import StandardScaler  # Importa escalador para estandarizar variables

# Cargar CSV
df = pd.read_csv("heart_attack_desease.csv")  # Lee el archivo CSV y lo almacena en un DataFrame llamado df

# Tabla estadística detallada
stats = pd.DataFrame({  # Crea un DataFrame con estadísticas por columna
    'Variable': df.columns,  # Nombres de las columnas
    'Tipo': [df[col].dtype for col in df.columns],  # Tipo de dato de cada columna
    'Media': [df[col].mean() for col in df.columns],  # Media (promedio)
    'Mediana': [df[col].median() for col in df.columns],  # Mediana
    'Moda': [df[col].mode().iloc[0] for col in df.columns],  # Moda (valor más frecuente)
    'Desv. Estándar': [df[col].std() for col in df.columns],  # Desviación estándar
    'Varianza': [df[col].var() for col in df.columns],  # Varianza
    'Min': [df[col].min() for col in df.columns],  # Valor mínimo
    '25%': [df[col].quantile(0.25) for col in df.columns],  # Primer cuartil
    '50%': [df[col].quantile(0.5) for col in df.columns],  # Mediana (de nuevo)
    '75%': [df[col].quantile(0.75) for col in df.columns],  # Tercer cuartil
    'Max': [df[col].max() for col in df.columns]  # Valor máximo
})
stats = stats.round(2)  # Redondea los valores a 2 decimales

# Mostrar tabla estadística
plt.figure(figsize=(18, 8))  # Crea figura de 18x8 pulgadas
plt.axis('off')  # Oculta los ejes
tbl = plt.table(cellText=stats.values, colLabels=stats.columns, loc='center', cellLoc='center')  # Crea tabla en gráfico
tbl.auto_set_font_size(False)  # Desactiva ajuste automático de fuente
tbl.set_fontsize(10)  # Establece tamaño de fuente a 10
tbl.scale(1.2, 1.5)  # Escala tabla horizontal y verticalmente
plt.title("Resumen Estadístico de las Variables", fontsize=16)  # Título del gráfico
plt.tight_layout()  # Ajusta el diseño para evitar solapamiento
plt.show()  # Muestra la tabla estadística

# Histograma + KDE y boxplot para cada variable
for col in df.columns:  # Recorre cada columna del DataFrame
    plt.figure(figsize=(14, 5))  # Crea una figura de 14x5 pulgadas

    # Histograma
    plt.subplot(1, 2, 1)  # Subgráfico 1 de 2 (a la izquierda)
    sns.histplot(df[col], kde=True, bins=30, color='cornflowerblue')  # Histograma con curva KDE
    plt.title(f'Histograma + KDE de {col}', fontsize=12)  # Título del histograma
    plt.xlabel(col)  # Etiqueta del eje X

    # Boxplot
    plt.subplot(1, 2, 2)  # Subgráfico 2 de 2 (a la derecha)
    sns.boxplot(x=df[col], color='salmon')  # Boxplot (diagrama de caja)
    plt.title(f'Boxplot de {col}', fontsize=12)  # Título del boxplot
    plt.xlabel(col)  # Etiqueta del eje X

    plt.tight_layout()  # Ajuste automático de diseño
    plt.show()  # Muestra el par de gráficos

# Matriz de correlación
plt.figure(figsize=(12, 10))  # Figura de 12x10 pulgadas
correlation = df.corr()  # Calcula la matriz de correlación entre todas las variables numéricas
sns.heatmap(correlation, annot=True, fmt=".2f", cmap='coolwarm', square=True)  # Muestra la matriz como mapa de calor
plt.title("Matriz de Correlación", fontsize=15)  # Título del gráfico
plt.tight_layout()  # Ajuste del diseño
plt.show()  # Muestra el heatmap

# Gráfico de pares (Pairplot) para visualizar relación entre variables
sns.pairplot(df, hue="target", diag_kind="kde", palette="husl")  # Gráfico de pares coloreado por clase "target"
plt.suptitle("Pairplot de Variables con Target", y=1.02, fontsize=16)  # Título del gráfico
plt.show()  # Muestra el gráfico de pares

# Análisis PCA para reducción de dimensionalidad
scaler = StandardScaler()  # Inicializa el estandarizador
X_scaled = scaler.fit_transform(df.drop('target', axis=1))  # Escala las variables (excepto target)

pca = PCA(n_components=len(df.columns)-1)  # Inicializa PCA con tantas componentes como variables originales menos una
pca.fit(X_scaled)  # Ajusta el PCA a los datos escalados
explained_variance = pca.explained_variance_ratio_  # Obtiene la varianza explicada por cada componente

# Gráfico de varianza explicada
plt.figure(figsize=(10, 6))  # Figura de 10x6 pulgadas
plt.plot(np.cumsum(explained_variance), marker='o', linestyle='--', color='green')  # Línea de varianza acumulada
plt.xlabel('Número de Componentes')  # Etiqueta del eje X
plt.ylabel('Varianza Acumulada')  # Etiqueta del eje Y
plt.title('PCA - Varianza Explicada Acumulada')  # Título del gráfico
plt.grid(True)  # Muestra la cuadrícula
plt.show()  # Muestra el gráfico de varianza acumulada

# Crear DataFrame con los componentes del PCA
pca_components = pd.DataFrame(pca.components_, columns=df.drop('target', axis=1).columns)  # Crea DataFrame con los pesos de cada variable en cada componente

# Tomar valor absoluto para visualizar la importancia relativa
abs_components = pca_components.abs()  # Toma el valor absoluto de los pesos para representar magnitud de influencia

# Mostrar como heatmap
plt.figure(figsize=(12, 6))  # Figura de 12x6 pulgadas
sns.heatmap(abs_components.T, annot=True, cmap="YlGnBu", cbar_kws={'label': 'Importancia (valor absoluto)'})  # Mapa de calor transpuesto con anotaciones
plt.title("Importancia de las variables en cada componente principal (PCA)")  # Título del gráfico
plt.xlabel("Componente Principal")  # Etiqueta del eje X
plt.ylabel("Variables originales")  # Etiqueta del eje Y
plt.tight_layout()  # Ajuste automático del diseño
plt.show()  # Muestra el heatmap final