# SIAF — Sistema Inteligente de Aislamiento de Fluidos
## Repositorio de Datos Preprocesados

**Universidad Tecnológica de León**  
**Materia:** Extracción de Conocimiento en Bases de Datos  
**Grupo:** IDGS-904  
**Equipo:** Juan Carlos López Ruiz, Cristopher Uriel Ramírez Olmos, Carlos Daniel Chavez Manriquez, Juan Daniel Nachez Hernández, Jiovani Pacheco López, Oscar Miguel Cáceres Loredo  
**Fecha:** Mayo 2026

---

## Contenido del repositorio

| Archivo | Descripción |
|---|---|
| `datos_siaf.csv` | Dataset original simulado de sensores IoT (300 registros) |
| `preprocesamiento_siaf.py` | Script Python con las 6 técnicas de limpieza aplicadas |
| `datos_siaf_preprocesados.csv` | Dataset limpio y normalizado listo para análisis |
| `grafico_anomalias_siaf.png` | Gráfico de anomalías detectadas por Isolation Forest |

---

## Descripción del dataset

El archivo `datos_siaf.csv` contiene lecturas simuladas de los sensores del sistema SIAF con las siguientes columnas:

- **id_registro** — Identificador único del registro
- **timestamp** — Fecha y hora de la lectura
- **seccion** — Tramo de la red (Centro_Norte, Sur_Baja, etc.)
- **flujo_lpm** — Flujo de agua en L/min (sensor YF-S201)
- **presion_bar** — Presión en bar (sensor de presión)
- **estado_valvula** — Estado de la válvula motorizada (Activa / Mantenimiento)
- **volumen_perdido_litros** — Volumen estimado perdido durante el evento
- **ciclos_valvula** — Ciclos acumulados de apertura/cierre
- **tipo_evento** — Clasificación del evento (Normal / Micro_fuga / Fuga_mayor)
- **gravedad** — Nivel de severidad (Baja / Media / Alta)

---

## Técnicas de preprocesamiento aplicadas

1. **Manejo de valores faltantes** — Sustitución con promedio (fillna)
2. **Eliminación de outliers de rango** — Filtrado por límites físicos del sensor
3. **Eliminación de duplicados** — Basada en timestamp y sección
4. **Codificación de variables categóricas** — Mapeo numérico
5. **Normalización** — StandardScaler (media=0, desviación=1)
6. **Detección de anomalías** — Isolation Forest (contamination=0.05)

---

## Herramientas utilizadas

- Python 3.x
- Pandas
- Scikit-learn
- Matplotlib

## Cómo ejecutar

```bash
pip install pandas scikit-learn matplotlib
python preprocesamiento_siaf.py
```
