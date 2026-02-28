# 📡 TelecomX – Predicción de Cancelación de Clientes (Churn)

> **Challenge de Ciencia de Datos | Parte 1 + Parte 2**  
> Análisis exploratorio, modelado predictivo y estrategias de retención para una empresa de telecomunicaciones.

---

## 📋 Descripción del Proyecto

Este proyecto forma parte del **Challenge TelecomX** de Alura LATAM, desarrollado en dos etapas:

- **Parte 1:** Análisis Exploratorio de Datos (EDA) — entender el comportamiento histórico del churn.
- **Parte 2:** Modelado Predictivo con Machine Learning — construir modelos capaces de anticipar qué clientes tienen mayor riesgo de cancelar sus servicios.

El objetivo final es proporcionar a TelecomX una herramienta basada en datos que le permita **anticiparse a la pérdida de clientes** e implementar estrategias de retención personalizadas.

---

## 📁 Estructura del Repositorio

```
telecomx-churn/
│
├── TelecomX_LATAM.ipynb          # Parte 1: EDA y análisis exploratorio
├── TelecomX_Parte2_ML.ipynb      # Parte 2: Modelado predictivo con ML
└── README.md                     # Este archivo
```

---

## 🗂️ Dataset

- **Fuente:** [TelecomX Data – GitHub Alura LATAM](https://raw.githubusercontent.com/ingridcristh/challenge2-data-science-LATAM/main/TelecomX_Data.json)
- **Formato:** JSON con columnas anidadas (customer, phone, internet, account)
- **Tamaño:** 7,267 clientes | 29 variables tras el desempaquetado
- **Variable objetivo:** `Churn` (1 = canceló, 0 = activo)
- **Churn rate base:** 25.72%

---

## 🔍 Parte 1 — Análisis Exploratorio de Datos (EDA)

### Proceso

**Extracción y Transformación**
- Carga del JSON desde la API de GitHub
- Desempaquetado de columnas anidadas (`customer`, `phone`, `internet`, `account`)
- Conversión de tipos de datos (`TotalCharges` a numérico)
- Tratamiento de valores nulos y duplicados
- Estandarización de la variable `Churn` (Yes/No → 1/0)

**Análisis Realizado**
- Distribución general del churn (25.72% de cancelaciones)
- Análisis demográfico: género, adultos mayores, pareja, dependientes
- Análisis de servicios: tipo de internet, líneas telefónicas, servicios adicionales
- Análisis de contrato: tipo, método de pago, facturación electrónica
- Análisis temporal: tenure (meses de contrato) y su relación con el churn
- Correlación entre variables numéricas y la cancelación

### Principales Hallazgos

| Factor | Churn Rate | Insight |
|--------|-----------|---------|
| Primeros 6 meses de contrato | 51.41% | Período crítico de fuga |
| Contrato Month-to-month | 41.32% | Sin compromiso = mayor riesgo |
| Pago con Electronic Check | 43.80% | Método manual = menor lealtad |
| Internet Fiber Optic | 40.56% | Problemas de satisfacción |
| Senior Citizens | 40.27% | Grupo vulnerable |
| Contrato Two-year | 2.75% | Mayor retención |
| 4+ años de contrato | 9.22% | Lealtad consolidada |

---

## 🤖 Parte 2 — Modelado Predictivo (Machine Learning)

### a) Preparación de los Datos

**Eliminación de columnas irrelevantes**  
Se eliminó `customerID` por ser un identificador único que no aporta valor predictivo y puede causar overfitting.

**Verificación de proporción de Churn**  
Se detectó un dataset desbalanceado: 74.28% No Churn vs 25.72% Churn (ratio 2.89:1). Se aplicó **Oversampling** (Random Oversampling) exclusivamente sobre el set de entrenamiento para evitar data leakage.

**Encoding de variables categóricas**  
- Variables binarias (Yes/No): codificadas como 1/0
- Variables con múltiples categorías (`Contract`, `PaymentMethod`, `InternetService`): **One-Hot Encoding**
- `gender`: Male=1, Female=0

**Normalización / Estandarización**  
Se crearon dos versiones del dataset con justificación técnica:
- **Con StandardScaler** (media=0, std=1): para Regresión Logística, sensible a la escala de los datos
- **Sin normalizar**: para Random Forest, basado en árboles y no sensible a la escala

**Balanceo de Clases**  
Oversampling aplicado solo al conjunto de entrenamiento para que el test refleje la distribución real del negocio.

### b) Correlación y Selección de Variables

- **Matriz de correlación** completa con heatmap de las top 12 variables
- **Análisis dirigido Tenure × Churn:** boxplot, histograma y tasa de churn por rango de meses
- **Análisis dirigido Gasto × Churn:** boxplot de MonthlyCharges y TotalCharges + scatter plot

### c) Modelos Entrenados

| | Regresión Logística | Random Forest |
|--|--------------------|----|
| **Normalización** | ✅ Sí (StandardScaler) | ❌ No necesaria |
| **Accuracy (test)** | ~0.75 | ~0.80 |
| **Precision** | ~0.49 | ~0.58 |
| **Recall** | ~0.80 | ~0.73 |
| **F1-Score** | ~0.61 | ~0.64 |
| **ROC-AUC** | ~0.84 | ~0.86 |

**¿Cuál modelo es mejor?**

- **Random Forest** obtiene mejor ROC-AUC y F1-Score → recomendado para scoring general
- **Regresión Logística** obtiene mayor Recall → más efectiva para capturar el máximo de churners posibles

En churn, el Recall es especialmente valioso: es más costoso no detectar a un cliente que se va (falso negativo) que alertar erróneamente a uno que se queda (falso positivo).

**Análisis de Overfitting / Underfitting**

- *Regresión Logística:* diferencia Train-Test < 3% → sin overfitting, buena generalización gracias a la regularización L2
- *Random Forest:* leve overfitting esperado (~10%) → mitigado con `max_depth=12` y `min_samples_leaf=5`

### d) Importancia de Variables

**Regresión Logística — Coeficientes:**
Los coeficientes positivos más altos corresponden a contratos month-to-month, Fiber Optic y Electronic Check. Los más negativos (protección) a contratos anuales y tenure alto.

**Random Forest — Gini Importance:**
Las variables con mayor importancia son `tenure`, `MonthlyCharges`, `TotalCharges`, seguidas por el tipo de contrato y el método de pago.

Ambos modelos coinciden en las mismas variables clave, lo que refuerza la solidez de los hallazgos.

---

## 💡 Conclusiones y Estrategias de Retención

### Factores principales que causan el churn

1. **Tenure bajo** — los primeros 6 meses son críticos (51% de cancelación)
2. **Contratos sin compromiso** — month-to-month tiene 41% de churn vs 2.75% en two-year
3. **Fiber Optic** — 40.56% de churn, posiblemente por baja satisfacción con la calidad
4. **Método de pago manual** — Electronic Check: 43.80% vs 14-16% en pagos automáticos
5. **Ausencia de servicios adicionales** — sin Tech Support ni Online Security: ~30% churn

### Estrategias propuestas

| Prioridad | Acción | Impacto Estimado |
|-----------|--------|-----------------|
| 🔴 1 | Score de riesgo predictivo en producción | Base para todas las acciones |
| 🔴 2 | Programa de onboarding intensivo (meses 0-6) | ~260 clientes retenidos/mes |
| 🟡 3 | Campaña de conversión a contratos anuales | -7-8 puntos en churn global |
| 🟡 4 | Migración a pagos automáticos (descuento 5%) | ~130 clientes retenidos/mes |
| 🟡 5 | Auditoría de calidad Fiber Optic | ~310 clientes retenidos/mes |
| 🟢 6 | Bundling de servicios adicionales | ~180 clientes retenidos/mes |

**Proyección conservadora (50% de adopción):**
- ~410 clientes retenidos por mes
- ~$320,000/mes en ingresos protegidos (ARPU ~$65)
- ROI anual estimado: ~$3.8M

---

## 🛠️ Tecnologías Utilizadas

- **Python 3**
- **Pandas** — manipulación y limpieza de datos
- **NumPy** — operaciones numéricas
- **Matplotlib / Seaborn** — visualización de datos
- **Scikit-learn** — modelado predictivo y evaluación
  - `LogisticRegression`
  - `RandomForestClassifier`
  - `StandardScaler`
  - `train_test_split`, `classification_report`, `roc_auc_score`

---

## ▶️ Cómo Ejecutar

1. Clona el repositorio:
```bash
git clone https://github.com/javito72/telecomx-churn.git
```

2. Abre los notebooks en [Google Colab](https://colab.research.google.com/) o Jupyter:
```
TelecomX_LATAM.ipynb      → Ejecutar primero (EDA)
TelecomX_Parte2_ML.ipynb  → Ejecutar segundo (ML)
```

3. Los notebooks se conectan directamente a la fuente de datos vía URL, no se requiere descarga previa del dataset.

---

## 👤 Autor: Christian Javier Lemos

Desarrollado como parte del **Challenge de Ciencia de Datos – Alura LATAM**  

---

*"Los datos no mienten: retener un cliente es siempre más barato que adquirir uno nuevo."*

