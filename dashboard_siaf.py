# ============================================================
# DASHBOARD SIAF - Sistema de Detección de Fugas
# Visualización de resultados con Matplotlib y Seaborn
# ============================================================

# --- Paso 1: Importar librerías ---
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

# --- Paso 2: Cargar datos ---
df = pd.read_csv("datos_siaf_preprocesados.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"], dayfirst=True)

print("=" * 60)
print("DASHBOARD SIAF - GENERANDO GRÁFICAS")
print("=" * 60)
print(f"Registros cargados: {len(df)}")

# ============================================================
# Configuración general de estilo
# ============================================================
sns.set_theme(style="whitegrid")
COLORES_EVENTO = {
    "Normal":     "#4CAF50",
    "Micro_fuga": "#FF9800",
    "Fuga_mayor": "#F44336"
}
COLORES_SECCION = [
    "#1976D2", "#388E3C", "#F57C00",
    "#7B1FA2", "#C62828", "#00796B"
]

# ============================================================
# GRÁFICA 1 — Distribución de eventos por tipo
# Gráfico de barras: compara la frecuencia de cada tipo de evento
# ============================================================
conteo_eventos = df["tipo_evento"].value_counts()

fig1, ax1 = plt.subplots(figsize=(7, 4))
barras = ax1.bar(
    conteo_eventos.index,
    conteo_eventos.values,
    color=[COLORES_EVENTO[e] for e in conteo_eventos.index],
    edgecolor="white", linewidth=1.2
)
for barra, valor in zip(barras, conteo_eventos.values):
    ax1.text(
        barra.get_x() + barra.get_width() / 2,
        barra.get_height() + 3,
        str(valor), ha="center", va="bottom", fontweight="bold"
    )
ax1.set_title("Distribución de eventos detectados por tipo", fontsize=13, fontweight="bold")
ax1.set_xlabel("Tipo de evento")
ax1.set_ylabel("Número de registros")
ax1.set_ylim(0, max(conteo_eventos.values) * 1.15)
plt.tight_layout()
plt.savefig("dashboard_g1_eventos.png", dpi=150, bbox_inches="tight")
plt.close()
print("✓ Gráfica 1 guardada: dashboard_g1_eventos.png")

# ============================================================
# GRÁFICA 2 — Presión promedio por sección
# Gráfico de barras horizontales: permite comparar el estado
# de presión en cada tramo de la red
# ============================================================
presion_seccion = df.groupby("seccion")["presion_bar"].mean().sort_values()

fig2, ax2 = plt.subplots(figsize=(8, 5))
barras_h = ax2.barh(
    presion_seccion.index,
    presion_seccion.values,
    color=COLORES_SECCION,
    edgecolor="white"
)
ax2.axvline(x=2.0, color="red", linestyle="--", linewidth=1.5, label="Umbral de riesgo (2 bar)")
for barra, valor in zip(barras_h, presion_seccion.values):
    ax2.text(
        valor + 0.05,
        barra.get_y() + barra.get_height() / 2,
        f"{valor:.2f} bar", va="center", fontsize=9
    )
ax2.set_title("Presión promedio por sección de la red", fontsize=13, fontweight="bold")
ax2.set_xlabel("Presión promedio (bar)")
ax2.set_ylabel("Sección")
ax2.legend()
plt.tight_layout()
plt.savefig("dashboard_g2_presion_seccion.png", dpi=150, bbox_inches="tight")
plt.close()
print("✓ Gráfica 2 guardada: dashboard_g2_presion_seccion.png")

# ============================================================
# GRÁFICA 3 — Flujo vs Presión coloreado por tipo de evento
# Diagrama de dispersión: muestra la relación entre ambas
# variables y cómo se separan los eventos normales de las fugas
# ============================================================
fig3, ax3 = plt.subplots(figsize=(8, 6))
for tipo, grupo in df.groupby("tipo_evento"):
    ax3.scatter(
        grupo["flujo_lpm"],
        grupo["presion_bar"],
        label=tipo,
        color=COLORES_EVENTO[tipo],
        alpha=0.7,
        s=40,
        edgecolors="white",
        linewidths=0.5
    )
ax3.axhline(y=2.0, color="red", linestyle="--", linewidth=1.5, label="Umbral de riesgo (2 bar)")
ax3.set_title("Flujo vs Presión por tipo de evento", fontsize=13, fontweight="bold")
ax3.set_xlabel("Flujo (L/min)")
ax3.set_ylabel("Presión (bar)")
ax3.legend()
plt.tight_layout()
plt.savefig("dashboard_g3_flujo_presion.png", dpi=150, bbox_inches="tight")
plt.close()
print("✓ Gráfica 3 guardada: dashboard_g3_flujo_presion.png")

# ============================================================
# GRÁFICA 4 — Evolución de presión en el tiempo por sección
# Gráfico de líneas: muestra la tendencia temporal de la
# presión en cada tramo de la red
# ============================================================
fig4, ax4 = plt.subplots(figsize=(12, 5))
for i, seccion in enumerate(df["seccion"].unique()):
    df_sec = df[df["seccion"] == seccion].sort_values("timestamp")
    ax4.plot(
        df_sec["timestamp"],
        df_sec["presion_bar"],
        label=seccion,
        color=COLORES_SECCION[i],
        linewidth=1.2,
        alpha=0.8
    )
ax4.axhline(y=2.0, color="red", linestyle="--", linewidth=1.5, label="Umbral de riesgo (2 bar)")
ax4.set_title("Evolución de presión en el tiempo por sección", fontsize=13, fontweight="bold")
ax4.set_xlabel("Fecha y hora")
ax4.set_ylabel("Presión (bar)")
ax4.legend(loc="upper right", fontsize=8)
plt.tight_layout()
plt.savefig("dashboard_g4_tendencia_presion.png", dpi=150, bbox_inches="tight")
plt.close()
print("✓ Gráfica 4 guardada: dashboard_g4_tendencia_presion.png")

# ============================================================
# GRÁFICA 5 — Mapa de calor: flujo promedio por sección y gravedad
# Heatmap: permite identificar rápidamente qué combinaciones
# de sección y gravedad presentan mayor consumo de flujo
# ============================================================
tabla_calor = df.pivot_table(
    values="flujo_lpm",
    index="seccion",
    columns="gravedad",
    aggfunc="mean"
)
fig5, ax5 = plt.subplots(figsize=(7, 5))
sns.heatmap(
    tabla_calor,
    annot=True,
    fmt=".2f",
    cmap="YlOrRd",
    linewidths=0.5,
    ax=ax5,
    cbar_kws={"label": "Flujo promedio (L/min)"}
)
ax5.set_title("Flujo promedio por sección y nivel de gravedad", fontsize=13, fontweight="bold")
ax5.set_xlabel("Nivel de gravedad")
ax5.set_ylabel("Sección")
plt.tight_layout()
plt.savefig("dashboard_g5_heatmap.png", dpi=150, bbox_inches="tight")
plt.close()
print("✓ Gráfica 5 guardada: dashboard_g5_heatmap.png")

# ============================================================
# GRÁFICA 6 — Volumen de agua perdida por sección
# Gráfico de barras: evidencia el impacto real de las fugas
# en cada tramo de la red en litros perdidos
# ============================================================
volumen_seccion = df.groupby("seccion")["volumen_perdido_litros"].sum().sort_values(ascending=False)

fig6, ax6 = plt.subplots(figsize=(8, 5))
barras6 = ax6.bar(
    volumen_seccion.index,
    volumen_seccion.values,
    color=COLORES_SECCION,
    edgecolor="white"
)
for barra, valor in zip(barras6, volumen_seccion.values):
    ax6.text(
        barra.get_x() + barra.get_width() / 2,
        barra.get_height() + 0.5,
        f"{valor:.1f} L",
        ha="center", va="bottom", fontsize=9, fontweight="bold"
    )
ax6.set_title("Volumen total de agua perdida por sección", fontsize=13, fontweight="bold")
ax6.set_xlabel("Sección")
ax6.set_ylabel("Volumen perdido (litros)")
ax6.tick_params(axis="x", rotation=15)
plt.tight_layout()
plt.savefig("dashboard_g6_volumen_perdido.png", dpi=150, bbox_inches="tight")
plt.close()
print("✓ Gráfica 6 guardada: dashboard_g6_volumen_perdido.png")

print("\n" + "=" * 60)
print("DASHBOARD GENERADO CORRECTAMENTE")
print("Archivos generados:")
print("  dashboard_g1_eventos.png")
print("  dashboard_g2_presion_seccion.png")
print("  dashboard_g3_flujo_presion.png")
print("  dashboard_g4_tendencia_presion.png")
print("  dashboard_g5_heatmap.png")
print("  dashboard_g6_volumen_perdido.png")
print("=" * 60)