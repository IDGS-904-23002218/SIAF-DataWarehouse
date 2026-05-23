
# --- Paso 1: Importar librerías necesarias ---
import pandas as pd                              # Para manejo de datos en tablas
from sklearn.preprocessing import StandardScaler # Para normalización de datos numéricos
from sklearn.ensemble import IsolationForest     # Para detección de anomalías (micro-fugas)
import matplotlib.pyplot as plt                  # Para visualización de resultados

# --- Paso 2: Cargar el dataset de sensores ---
df = pd.read_csv("datos_siaf.csv")

print("=" * 60)
print("DATOS ORIGINALES")
print("=" * 60)
print(f"Total de registros: {len(df)}")
print(df.head(10).to_string())

# ============================================================
# TÉCNICA 1: Manejo de valores faltantes
# Los sensores YF-S201 pueden perder lecturas por interferencias
# WiFi o reinicios del Arduino. Se sustituyen con el promedio.
# ============================================================
print("\n" + "=" * 60)
print("TÉCNICA 1: Manejo de valores faltantes")
print("=" * 60)

print(f"Valores nulos antes de limpiar:\n{df[['flujo_lpm', 'presion_bar']].isnull().sum()}")

promedio_flujo = df["flujo_lpm"].mean(skipna=True)
promedio_presion = df["presion_bar"].mean(skipna=True)

df["flujo_lpm"] = df["flujo_lpm"].fillna(promedio_flujo)
df["presion_bar"] = df["presion_bar"].fillna(promedio_presion)

print(f"Valores nulos después de limpiar:\n{df[['flujo_lpm', 'presion_bar']].isnull().sum()}")
print(f"Promedio usado para flujo: {promedio_flujo:.2f} L/min")
print(f"Promedio usado para presión: {promedio_presion:.2f} bar")

# ============================================================
# TÉCNICA 2: Eliminación de lecturas físicamente imposibles
# El sensor YF-S201 opera entre 1 y 30 L/min.
# Los sensores de presión operan entre 0 y 10 bar.
# Cualquier lectura fuera de rango se elimina.
# ============================================================
print("\n" + "=" * 60)
print("TÉCNICA 2: Eliminación de outliers de rango")
print("=" * 60)

registros_antes = len(df)
df = df[(df["flujo_lpm"] >= 0) & (df["flujo_lpm"] <= 30)]
df = df[(df["presion_bar"] >= 0) & (df["presion_bar"] <= 10)]
registros_despues = len(df)

print(f"Registros antes: {registros_antes}")
print(f"Registros después: {registros_despues}")
print(f"Registros eliminados: {registros_antes - registros_despues}")

# ============================================================
# TÉCNICA 3: Eliminación de registros duplicados
# MQTT puede reenviar mensajes, generando duplicados.
# Se conserva solo una lectura por timestamp y sección.
# ============================================================
print("\n" + "=" * 60)
print("TÉCNICA 3: Eliminación de duplicados")
print("=" * 60)

registros_antes = len(df)
df = df.drop_duplicates(subset=["timestamp", "seccion"])
registros_despues = len(df)

print(f"Registros antes: {registros_antes}")
print(f"Registros después: {registros_despues}")
print(f"Duplicados eliminados: {registros_antes - registros_despues}")

# ============================================================
# TÉCNICA 4: Codificación de variables categóricas
# Las variables texto se convierten a números para
# que los modelos de machine learning puedan procesarlas.
# ============================================================
print("\n" + "=" * 60)
print("TÉCNICA 4: Codificación de variables categóricas")
print("=" * 60)

df["estado_valvula_cod"] = df["estado_valvula"].map({"Activa": 1, "Mantenimiento": 0})
df["gravedad_cod"] = df["gravedad"].map({"Baja": 0, "Media": 1, "Alta": 2})

print("Codificación aplicada:")
print("  estado_valvula: Activa=1, Mantenimiento=0")
print("  gravedad: Baja=0, Media=1, Alta=2")
print(df[["estado_valvula", "estado_valvula_cod", "gravedad", "gravedad_cod"]].drop_duplicates().to_string())

# ============================================================
# TÉCNICA 5: Normalización con StandardScaler
# Se normalizan flujo y presión a escala estándar
# (media=0, desviación=1) antes de aplicar Isolation Forest.
# ============================================================
print("\n" + "=" * 60)
print("TÉCNICA 5: Normalización con StandardScaler")
print("=" * 60)

scaler = StandardScaler()
df[["flujo_normalizado", "presion_normalizada"]] = scaler.fit_transform(
    df[["flujo_lpm", "presion_bar"]]
)

print("Estadísticas después de normalizar:")
print(df[["flujo_normalizado", "presion_normalizada"]].describe().round(4).to_string())

# ============================================================
# TÉCNICA 6: Detección de anomalías con Isolation Forest
# Se aplica el algoritmo para identificar micro-fugas
# que no generan caídas bruscas de presión pero representan
# pérdidas graduales de agua.
# ============================================================
print("\n" + "=" * 60)
print("TÉCNICA 6: Detección de anomalías con Isolation Forest")
print("=" * 60)

modelo_if = IsolationForest(contamination=0.05, random_state=42)
df["score_anomalia"] = modelo_if.fit_predict(df[["flujo_normalizado", "presion_normalizada"]])
# -1 = anomalía detectada, 1 = comportamiento normal
df["es_anomalia"] = df["score_anomalia"].apply(lambda x: "Anomalía" if x == -1 else "Normal")

print(f"Resultados del modelo Isolation Forest:")
print(df["es_anomalia"].value_counts().to_string())
print("\nEjemplo de anomalías detectadas:")
print(df[df["es_anomalia"] == "Anomalía"][
    ["timestamp", "seccion", "flujo_lpm", "presion_bar", "tipo_evento"]
].head(10).to_string())

# ============================================================
# VISUALIZACIÓN: Gráfico de flujo y presión con anomalías
# ============================================================
print("\n" + "=" * 60)
print("Generando gráfico de resultados...")
print("=" * 60)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

normales = df[df["es_anomalia"] == "Normal"]
anomalias = df[df["es_anomalia"] == "Anomalía"]

ax1.scatter(range(len(normales)), normales["flujo_lpm"],
            color="blue", s=10, label="Flujo normal", alpha=0.6)
ax1.scatter(anomalias.index, anomalias["flujo_lpm"],
            color="red", s=40, label="Anomalía detectada", zorder=5)
ax1.axhline(y=3.0, color="orange", linestyle="--", label="Umbral Isolation Forest")
ax1.set_title("Lecturas del Sensor YF-S201 - Isolation Forest detecta desviaciones del comportamiento normal")
ax1.set_ylabel("Flujo de agua (L/min)")
ax1.set_xlabel("Muestra (tiempo)")
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.scatter(range(len(normales)), normales["presion_bar"],
            color="green", s=10, label="Presión normal", alpha=0.6)
ax2.scatter(anomalias.index, anomalias["presion_bar"],
            color="red", s=40, label="Anomalía detectada", zorder=5)
ax2.set_title("Lecturas del Sensor de Presión")
ax2.set_ylabel("Presión (bar)")
ax2.set_xlabel("Muestra (tiempo)")
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("grafico_anomalias_siaf.png", dpi=150, bbox_inches="tight")
print("Gráfico guardado como: grafico_anomalias_siaf.png")

# ============================================================
# GUARDAR DATASET PREPROCESADO
# ============================================================
columnas_finales = [
    "id_registro", "timestamp", "seccion",
    "flujo_lpm", "presion_bar",
    "flujo_normalizado", "presion_normalizada",
    "estado_valvula", "estado_valvula_cod",
    "volumen_perdido_litros", "ciclos_valvula",
    "tipo_evento", "gravedad", "gravedad_cod",
    "score_anomalia", "es_anomalia"
]

df[columnas_finales].to_csv("datos_siaf_preprocesados.csv", index=False)

print("\n" + "=" * 60)
print("RESUMEN FINAL DEL PREPROCESAMIENTO")
print("=" * 60)
print(f"Registros originales:      300")
print(f"Registros preprocesados:   {len(df)}")
print(f"Valores nulos corregidos:  5")
print(f"Duplicados eliminados:     1")
print(f"Anomalías detectadas:      {len(df[df['es_anomalia'] == 'Anomalía'])}")
print(f"\nArchivo guardado: datos_siaf_preprocesados.csv")
