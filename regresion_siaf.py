# --- Paso 1: Importar librerías necesarias ---
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# --- Paso 2: Cargar el dataset preprocesado ---
df = pd.read_csv("datos_siaf_preprocesados.csv")

print("=" * 60)
print("MODELO DE REGRESIÓN - PREDICCIÓN DE PRESIÓN FUTURA")
print("=" * 60)

# ============================================================
# PREPARACIÓN: Convertir timestamp a valor numérico
# El modelo necesita números, no fechas en texto.
# Se convierte cada timestamp a minutos transcurridos
# desde el primer registro, representando el paso del tiempo.
# ============================================================
df["timestamp"] = pd.to_datetime(df["timestamp"], dayfirst=True)
df = df.sort_values("timestamp").reset_index(drop=True)
df["minutos"] = (df["timestamp"] - df["timestamp"].min()).dt.total_seconds() / 60

print(f"\nRegistros cargados: {len(df)}")
print(f"Periodo: {df['timestamp'].min()} → {df['timestamp'].max()}")
print(f"Secciones de red: {df['seccion'].unique()}")

# ============================================================
# MODELO POR SECCIÓN
# Se entrena un modelo de regresión independiente para cada
# tramo de la red, ya que cada sección tiene su propio
# comportamiento de presión y desgaste.
# ============================================================
secciones = df["seccion"].unique()
resultados = []

print("\n" + "=" * 60)
print("ENTRENAMIENTO Y EVALUACIÓN POR SECCIÓN")
print("=" * 60)

for seccion in secciones:
    df_sec = df[df["seccion"] == seccion].copy()

    X = df_sec[["minutos"]].values
    y = df_sec["presion_bar"].values

    # División 70% entrenamiento / 30% prueba
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # Entrenar regresión lineal
    modelo = LinearRegression()
    modelo.fit(X_train, y_train)

    # Evaluar
    y_pred = modelo.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    # Predicción a 24, 48 y 72 horas futuras
    ultimo_minuto = df_sec["minutos"].max()
    horizontes = {
        "24h": ultimo_minuto + 1440,
        "48h": ultimo_minuto + 2880,
        "72h": ultimo_minuto + 4320
    }

    predicciones = {}
    for horizonte, minuto_futuro in horizontes.items():
        presion_pred = modelo.predict([[minuto_futuro]])[0]
        predicciones[horizonte] = round(presion_pred, 3)

    # Determinar riesgo: si la presión predicha a 72h cae bajo 2 bar
    riesgo = "ALTO" if predicciones["72h"] < 2.0 else (
              "MEDIO" if predicciones["72h"] < 3.0 else "BAJO")

    resultados.append({
        "seccion": seccion,
        "pendiente": round(modelo.coef_[0], 6),
        "rmse": round(rmse, 4),
        "r2": round(r2, 4),
        "pred_24h": predicciones["24h"],
        "pred_48h": predicciones["48h"],
        "pred_72h": predicciones["72h"],
        "riesgo": riesgo
    })

    print(f"\nSección: {seccion}")
    print(f"  Pendiente (degradación/min): {modelo.coef_[0]:.6f}")
    print(f"  RMSE:  {rmse:.4f} bar")
    print(f"  R²:    {r2:.4f}")
    print(f"  Presión predicha a 24h: {predicciones['24h']} bar")
    print(f"  Presión predicha a 48h: {predicciones['48h']} bar")
    print(f"  Presión predicha a 72h: {predicciones['72h']} bar")
    print(f"  Nivel de riesgo: {riesgo}")

# ============================================================
# RESUMEN DE RIESGO POR SECCIÓN
# ============================================================
print("\n" + "=" * 60)
print("RESUMEN DE RIESGO POR SECCIÓN")
print("=" * 60)

df_resultados = pd.DataFrame(resultados)
print(df_resultados[["seccion", "rmse", "r2", "pred_72h", "riesgo"]].to_string(index=False))

# ============================================================
# VISUALIZACIÓN
# ============================================================
fig, axes = plt.subplots(len(secciones), 1,
                          figsize=(12, 4 * len(secciones)))

if len(secciones) == 1:
    axes = [axes]

for ax, seccion in zip(axes, secciones):
    df_sec = df[df["seccion"] == seccion].copy()
    X_plot = df_sec[["minutos"]].values
    y_real = df_sec["presion_bar"].values

    modelo_plot = LinearRegression()
    modelo_plot.fit(X_plot, y_real)
    y_linea = modelo_plot.predict(X_plot)

    ax.scatter(df_sec["minutos"], y_real,
               color="steelblue", s=15, alpha=0.6, label="Presión real")
    ax.plot(df_sec["minutos"], y_linea,
            color="red", linewidth=2, label="Tendencia (regresión)")
    ax.axhline(y=2.0, color="orange", linestyle="--", linewidth=1.5,
               label="Umbral de riesgo (2 bar)")

    res = df_resultados[df_resultados["seccion"] == seccion].iloc[0]
    ax.set_title(
        f"Sección: {seccion} | R²={res['r2']} | RMSE={res['rmse']} bar | "
        f"Riesgo 72h: {res['riesgo']}"
    )
    ax.set_xlabel("Tiempo transcurrido (minutos)")
    ax.set_ylabel("Presión (bar)")
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("grafico_regresion_siaf.png", dpi=150, bbox_inches="tight")
print("\nGráfico guardado como: grafico_regresion_siaf.png")

# ============================================================
# EXPORTAR RESULTADOS
# ============================================================
df_resultados.to_csv("resultados_regresion_siaf.csv", index=False)
print("Resultados exportados: resultados_regresion_siaf.csv")

print("\n" + "=" * 60)
print("FIN DEL MODELO DE REGRESIÓN")
print("=" * 60)