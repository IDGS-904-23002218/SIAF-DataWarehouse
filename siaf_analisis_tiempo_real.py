"""
SIAF - Análisis en tiempo real sobre datos REALES de la base de datos
======================================================================
Versión adaptada de preprocesamiento_siaf.py + clustering_siaf.py +
regresion_siaf.py (materia de Extracción de Conocimiento en Bases de
Datos), para correr sobre la tabla `lecturas` real (poblada por el
firmware SIAF_ESP32_v3.ino), en vez del dataset simulado de la materia.

Diferencias clave respecto a los scripts originales de la materia:
  1. Lee de MySQL (Railway) en vez de un CSV estático.
  2. Usa los nombres de sección ACTUALES del prototipo físico
     (Entrada, Tramo_Izquierdo, Tramo_Derecho) en vez de los antiguos
     (Centro_Norte, Sur_Baja, etc.)
  3. Los datos reales NO vienen pre-etiquetados con tipo_evento/gravedad
     (a diferencia del dataset simulado de la materia) -- este script
     los INFIERE a partir del resultado de Isolation Forest / K-means,
     con una heurística simple documentada abajo.
  4. Escribe los resultados directamente en las tablas `anomalias` y
     `predicciones_riesgo`, evitando reprocesar lecturas ya analizadas.

Pensado para ejecutarse periódicamente (cron, tarea programada, o
manualmente), no en cada request de la app.

Requisitos: pip install -r requirements_analisis.txt
Variables de entorno esperadas (mismas que usa conexion.php en Railway):
  MYSQLHOST, MYSQLDATABASE, MYSQLUSER, MYSQLPASSWORD, MYSQLPORT
"""

import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import mysql.connector
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression

# ============================================================
# Secciones que SÍ tienen sensores propios (flujo y/o presión).
# "Salida" queda fuera a propósito: no tiene sensor dedicado en el
# prototipo físico, solo una válvula y un codo de salida.
# ============================================================
SECCIONES_CON_SENSOR = ["Entrada", "Tramo_Izquierdo", "Tramo_Derecho", "Parte_Abajo"]


def conectar_bd():
    """Conexión a MySQL usando las mismas variables de entorno que conexion.php."""
    return mysql.connector.connect(
        host=os.environ["MYSQLHOST"],
        database=os.environ["MYSQLDATABASE"],
        user=os.environ["MYSQLUSER"],
        password=os.environ["MYSQLPASSWORD"],
        port=int(os.environ.get("MYSQLPORT", 3306)),
    )


# ============================================================
# PASO 1: Cargar lecturas que aún no han sido analizadas
# (evita reprocesar lo mismo cada vez que corre el script)
# ============================================================
def cargar_lecturas_pendientes(conexion):
    query = """
        SELECT l.id, l.timestamp, l.seccion, l.flujo_lpm, l.presion_bar,
               l.estado_valvula, l.volumen_perdido_litros, l.ciclos_valvula
        FROM lecturas l
        LEFT JOIN anomalias a ON a.id_lectura = l.id
        WHERE a.id IS NULL
          AND l.seccion IN ({})
        ORDER BY l.timestamp ASC
    """.format(",".join(["%s"] * len(SECCIONES_CON_SENSOR)))

    df = pd.read_sql(query, conexion, params=SECCIONES_CON_SENSOR)
    return df


# ============================================================
# PASO 2: Preprocesamiento (igual espíritu que preprocesamiento_siaf.py,
# adaptado: sin columnas de etiquetas que no existen en datos reales)
# ============================================================
def preprocesar(df):
    # Manejo de nulos
    for columna in ["flujo_lpm", "presion_bar"]:
        if df[columna].isnull().any():
            promedio = df[columna].mean(skipna=True)
            df[columna] = df[columna].fillna(promedio)

    # Eliminar lecturas fuera de rango físico posible
    df = df[(df["flujo_lpm"] >= 0) & (df["flujo_lpm"] <= 30)]
    df = df[(df["presion_bar"] >= 0) & (df["presion_bar"] <= 10)]

    # Eliminar duplicados exactos (mismo timestamp y sección)
    df = df.drop_duplicates(subset=["timestamp", "seccion"])

    if len(df) < 10:
        return df, None  # muy pocos datos para normalizar/entrenar de forma confiable

    scaler = StandardScaler()
    df[["flujo_normalizado", "presion_normalizada"]] = scaler.fit_transform(
        df[["flujo_lpm", "presion_bar"]]
    )
    return df, scaler


# ============================================================
# PASO 3: Isolation Forest + K-means (igual que en la materia,
# pero sin comparar contra etiquetas preexistentes, porque no existen
# en datos reales)
# ============================================================
def detectar_anomalias(df):
    modelo_if = IsolationForest(contamination=0.05, random_state=42)
    df["score_anomalia"] = modelo_if.fit_predict(
        df[["flujo_normalizado", "presion_normalizada"]]
    )
    df["es_anomalia"] = df["score_anomalia"].apply(
        lambda x: "Anomalia" if x == -1 else "Normal"
    )

    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    df["cluster"] = kmeans.fit_predict(
        df[["flujo_normalizado", "presion_normalizada"]]
    )

    return df


# ============================================================
# PASO 4: Inferir tipo_evento y gravedad (heurística, ya que datos
# reales no vienen pre-etiquetados como el dataset simulado de la materia)
# ============================================================
def inferir_tipo_evento_y_gravedad(fila):
    if fila["es_anomalia"] == "Normal":
        return "Normal", "Baja"

    # Si el flujo se desvió mucho más que la presión, sugiere fuga
    # (más agua de la esperada); si la presión se desvió más, sugiere
    # caída de presión sin explicación de flujo (posible fuga lenta
    # o problema de sensor).
    desviacion_flujo = abs(fila["flujo_normalizado"])
    desviacion_presion = abs(fila["presion_normalizada"])

    if desviacion_flujo > desviacion_presion * 1.5:
        tipo_evento = "Fuga_probable"
    elif desviacion_presion > desviacion_flujo * 1.5:
        tipo_evento = "Caida_presion"
    else:
        tipo_evento = "Anomalia_general"

    # Gravedad según qué tan extrema es la desviación combinada
    desviacion_total = desviacion_flujo + desviacion_presion
    if desviacion_total > 4.0:
        gravedad = "Alta"
    elif desviacion_total > 2.0:
        gravedad = "Media"
    else:
        gravedad = "Baja"

    return tipo_evento, gravedad


# ============================================================
# PASO 5: Guardar resultados en la tabla `anomalias`
# ============================================================
def guardar_anomalias(conexion, df):
    cursor = conexion.cursor()
    filas_insertadas = 0

    for _, fila in df.iterrows():
        tipo_evento, gravedad = inferir_tipo_evento_y_gravedad(fila)

        cursor.execute(
            """
            INSERT INTO anomalias (id_lectura, es_anomalia, cluster, tipo_evento, gravedad)
            VALUES (%s, %s, %s, %s, %s)
            """,
            (
                int(fila["id"]),
                fila["es_anomalia"],
                int(fila["cluster"]),
                tipo_evento,
                gravedad,
            ),
        )
        filas_insertadas += 1

    conexion.commit()
    cursor.close()
    return filas_insertadas


# ============================================================
# PASO 6: Regresión de presión por sección -> predicciones_riesgo
# (mismo enfoque que regresion_siaf.py, con nombres de sección reales)
# ============================================================
def calcular_predicciones_riesgo(conexion):
    query = """
        SELECT timestamp, seccion, presion_bar
        FROM lecturas
        WHERE seccion IN ({})
        ORDER BY timestamp ASC
    """.format(",".join(["%s"] * len(SECCIONES_CON_SENSOR)))

    df = pd.read_sql(query, conexion, params=SECCIONES_CON_SENSOR)

    if df.empty:
        print("No hay lecturas suficientes para calcular predicciones de riesgo todavía.")
        return

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    cursor = conexion.cursor()

    for seccion in SECCIONES_CON_SENSOR:
        df_sec = df[df["seccion"] == seccion].copy()

        # Se requiere un mínimo de puntos para que la regresión tenga sentido
        if len(df_sec) < 10:
            print(f"Sección {seccion}: aún no hay suficientes lecturas para predecir (mínimo 10).")
            continue

        df_sec["minutos"] = (
            df_sec["timestamp"] - df_sec["timestamp"].min()
        ).dt.total_seconds() / 60

        X = df_sec[["minutos"]].values
        y = df_sec["presion_bar"].values

        modelo = LinearRegression()
        modelo.fit(X, y)

        ultimo_minuto = df_sec["minutos"].max()
        pred_24h = float(modelo.predict([[ultimo_minuto + 1440]])[0])
        pred_48h = float(modelo.predict([[ultimo_minuto + 2880]])[0])
        pred_72h = float(modelo.predict([[ultimo_minuto + 4320]])[0])

        riesgo = "ALTO" if pred_72h < 2.0 else ("MEDIO" if pred_72h < 3.0 else "BAJO")

        cursor.execute(
            """
            INSERT INTO predicciones_riesgo (seccion, fecha_calculo, pred_24h, pred_48h, pred_72h, riesgo)
            VALUES (%s, %s, %s, %s, %s, %s)
            """,
            (seccion, datetime.now(), round(pred_24h, 3), round(pred_48h, 3), round(pred_72h, 3), riesgo),
        )

        print(f"Sección {seccion}: riesgo={riesgo} | pred_72h={round(pred_72h, 3)} bar")

    conexion.commit()
    cursor.close()


# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 60)
    print("SIAF - Análisis en tiempo real sobre datos reales")
    print("=" * 60)

    conexion = conectar_bd()

    # --- Detección de anomalías sobre lecturas nuevas ---
    df_pendientes = cargar_lecturas_pendientes(conexion)
    print(f"\nLecturas nuevas por analizar: {len(df_pendientes)}")

    if len(df_pendientes) >= 10:
        df_procesado, _ = preprocesar(df_pendientes)
        if df_procesado is not None and len(df_procesado) >= 10:
            df_procesado = detectar_anomalias(df_procesado)
            insertadas = guardar_anomalias(conexion, df_procesado)
            print(f"Anomalías/registros guardados en tabla 'anomalias': {insertadas}")
        else:
            print("No hay suficientes lecturas válidas tras el preprocesamiento (mínimo 10).")
    else:
        print("Aún no hay suficientes lecturas nuevas para análisis de anomalías (mínimo 10).")
        print("Esto es normal si el ESP32 apenas empezó a reportar a 'lecturas'.")

    # --- Predicción de riesgo por sección ---
    print("\nCalculando predicciones de riesgo por sección...")
    calcular_predicciones_riesgo(conexion)

    conexion.close()
    print("\nAnálisis completado.")


if __name__ == "__main__":
    try:
        main()
    except mysql.connector.Error as e:
        print(f"Error de conexión a la base de datos: {e}", file=sys.stderr)
        sys.exit(1)