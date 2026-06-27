# --- Paso 1: Importar librerías necesarias ---
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# --- Paso 2: Cargar el dataset preprocesado ---
df = pd.read_csv("datos_siaf_preprocesados.csv")

print("=" * 60)
print("ANÁLISIS NO SUPERVISADO - K-MEANS + PCA")
print("=" * 60)
print(f"\nRegistros cargados: {len(df)}")
print(f"Distribución real de eventos:\n{df['tipo_evento'].value_counts()}")

# ============================================================
# Paso 3: Seleccionar variables y normalizar
# Usamos flujo y presión normalizados (ya generados en el
# preprocesamiento) para mantener consistencia con Isolation Forest
# ============================================================
X = df[["flujo_normalizado", "presion_normalizada"]].values

# ============================================================
# Paso 4: Aplicar K-means con 2 clusters
# (comportamiento normal vs comportamiento atípico)
# ============================================================
kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
df["cluster"] = kmeans.fit_predict(X)

# Identificar cuál cluster es el minoritario (probable anómalo)
conteo_clusters = df["cluster"].value_counts()
cluster_minoritario = conteo_clusters.idxmin()
print(f"\nDistribución de clusters K-means:\n{conteo_clusters}")
print(f"Cluster minoritario (probable atípico): {cluster_minoritario}")

# ============================================================
# Paso 5: Comparar clusters de K-means contra las anomalías
# detectadas previamente por Isolation Forest
# ============================================================
df["es_cluster_atipico"] = df["cluster"] == cluster_minoritario
df["es_anomalia_if"] = df["es_anomalia"] == "Anomalia"

coincidencias = (df["es_cluster_atipico"] == df["es_anomalia_if"]).sum()
porcentaje_coincidencia = (coincidencias / len(df)) * 100

print(f"\nCoincidencia entre cluster atípico (K-means) y anomalía (Isolation Forest): "
      f"{porcentaje_coincidencia:.2f}%")

tabla_cruzada = pd.crosstab(df["cluster"], df["es_anomalia"])
print(f"\nTabla cruzada cluster vs. clasificación Isolation Forest:\n{tabla_cruzada}")

# ============================================================
# Paso 5b: Métricas de evaluación del clustering
# ============================================================
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

silhouette = silhouette_score(X, df["cluster"])
davies = davies_bouldin_score(X, df["cluster"])
calinski = calinski_harabasz_score(X, df["cluster"])

print("\n" + "=" * 60)
print("MÉTRICAS DE EVALUACIÓN DEL CLUSTERING")
print("=" * 60)
print(f"Silhouette Score:        {silhouette:.4f}  (rango -1 a 1, mayor es mejor)")
print(f"Davies-Bouldin Index:    {davies:.4f}  (menor es mejor)")
print(f"Calinski-Harabasz Index: {calinski:.4f} (mayor es mejor)")

# ============================================================
# Paso 6: Graficar clusters en el espacio original
# (flujo vs presión normalizados)
# ============================================================
plt.figure(figsize=(8, 6))
plt.scatter(df["flujo_normalizado"], df["presion_normalizada"],
            c=df["cluster"], cmap="rainbow", s=25, alpha=0.7)
plt.xlabel("Flujo normalizado")
plt.ylabel("Presión normalizada")
plt.title("Clusters K-means sobre datos de SIAF (flujo vs presión)")
plt.colorbar(label="Cluster")
plt.grid(True, alpha=0.3)
plt.savefig("grafico_kmeans_siaf.png", dpi=150, bbox_inches="tight")
print("\nGráfico guardado como: grafico_kmeans_siaf.png")

# ============================================================
# Paso 7: Reducción de dimensionalidad con PCA
# Se usan más variables para una proyección más representativa
# ============================================================
columnas_pca = ["flujo_normalizado", "presion_normalizada",
                 "ciclos_valvula", "volumen_perdido_litros"]

scaler = StandardScaler()
X_pca_input = scaler.fit_transform(df[columnas_pca])

pca = PCA(n_components=2)
componentes = pca.fit_transform(X_pca_input)

df["pca_1"] = componentes[:, 0]
df["pca_2"] = componentes[:, 1]

varianza_explicada = pca.explained_variance_ratio_
print(f"\nVarianza explicada por componente: {varianza_explicada}")
print(f"Varianza total explicada (2 componentes): {sum(varianza_explicada):.4f}")

# ============================================================
# Paso 8: Graficar resultado de PCA coloreado por cluster
# ============================================================
plt.figure(figsize=(8, 6))
scatter = plt.scatter(df["pca_1"], df["pca_2"],
                       c=df["cluster"], cmap="rainbow", s=25, alpha=0.7)
plt.xlabel(f"Componente principal 1 ({varianza_explicada[0]*100:.1f}% varianza)")
plt.ylabel(f"Componente principal 2 ({varianza_explicada[1]*100:.1f}% varianza)")
plt.title("Proyección PCA de clusters K-means - SIAF")
plt.colorbar(scatter, label="Cluster")
plt.grid(True, alpha=0.3)
plt.savefig("grafico_pca_siaf.png", dpi=150, bbox_inches="tight")
print("Gráfico guardado como: grafico_pca_siaf.png")

# ============================================================
# Paso 9: Exportar resultados
# ============================================================
df[["id_registro", "seccion", "flujo_lpm", "presion_bar", "cluster",
    "es_anomalia", "tipo_evento"]].to_csv("resultados_clustering_siaf.csv", index=False)
print("Resultados exportados: resultados_clustering_siaf.csv")

print("\n" + "=" * 60)
print("FIN DEL ANÁLISIS NO SUPERVISADO")
print("=" * 60)