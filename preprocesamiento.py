import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Cargar CSV
df = pd.read_csv("heart_attack_desease.csv")

# Tabla estadística detallada
stats = pd.DataFrame({
    'Variable': df.columns,
    'Tipo': [df[col].dtype for col in df.columns],
    'Media': [df[col].mean() for col in df.columns],
    'Mediana': [df[col].median() for col in df.columns],
    'Moda': [df[col].mode().iloc[0] for col in df.columns],
    'Desv. Estándar': [df[col].std() for col in df.columns],
    'Varianza': [df[col].var() for col in df.columns],
    'Min': [df[col].min() for col in df.columns],
    '25%': [df[col].quantile(0.25) for col in df.columns],
    '50%': [df[col].quantile(0.5) for col in df.columns],
    '75%': [df[col].quantile(0.75) for col in df.columns],
    'Max': [df[col].max() for col in df.columns]
})
stats = stats.round(2)

# Mostrar tabla estadística
plt.figure(figsize=(18, 8))
plt.axis('off')
tbl = plt.table(cellText=stats.values, colLabels=stats.columns, loc='center', cellLoc='center')
tbl.auto_set_font_size(False)
tbl.set_fontsize(10)
tbl.scale(1.2, 1.5)
plt.title("Resumen Estadístico de las Variables", fontsize=16)
plt.tight_layout()
plt.show()

# Histograma + KDE y boxplot para cada variable
for col in df.columns:
    plt.figure(figsize=(14, 5))

    # Histograma
    plt.subplot(1, 2, 1)
    sns.histplot(df[col], kde=True, bins=30, color='cornflowerblue')
    plt.title(f'Histograma + KDE de {col}', fontsize=12)
    plt.xlabel(col)

    # Boxplot
    plt.subplot(1, 2, 2)
    sns.boxplot(x=df[col], color='salmon')
    plt.title(f'Boxplot de {col}', fontsize=12)
    plt.xlabel(col)

    plt.tight_layout()
    plt.show()

# Matriz de correlación
plt.figure(figsize=(12, 10))
correlation = df.corr()
sns.heatmap(correlation, annot=True, fmt=".2f", cmap='coolwarm', square=True)
plt.title("Matriz de Correlación", fontsize=15)
plt.tight_layout()
plt.show()

# Gráfico de pares (Pairplot) para visualizar relación entre variables
sns.pairplot(df, hue="target", diag_kind="kde", palette="husl")
plt.suptitle("Pairplot de Variables con Target", y=1.02, fontsize=16)
plt.show()

# Análisis PCA para reducción de dimensionalidad
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df.drop('target', axis=1))  # Excluimos target

pca = PCA(n_components=len(df.columns)-1)
pca.fit(X_scaled)
explained_variance = pca.explained_variance_ratio_

# Gráfico de varianza explicada
plt.figure(figsize=(10, 6))
plt.plot(np.cumsum(explained_variance), marker='o', linestyle='--', color='green')
plt.xlabel('Número de Componentes')
plt.ylabel('Varianza Acumulada')
plt.title('PCA - Varianza Explicada Acumulada')
plt.grid(True)
plt.show()

# Crear DataFrame con los componentes del PCA
pca_components = pd.DataFrame(pca.components_, columns=df.drop('target', axis=1).columns)

# Tomar valor absoluto para visualizar la importancia relativa
abs_components = pca_components.abs()

# Mostrar como heatmap
plt.figure(figsize=(12, 6))
sns.heatmap(abs_components.T, annot=True, cmap="YlGnBu", cbar_kws={'label': 'Importancia (valor absoluto)'})
plt.title("Importancia de las variables en cada componente principal (PCA)")
plt.xlabel("Componente Principal")
plt.ylabel("Variables originales")
plt.tight_layout()
plt.show()
