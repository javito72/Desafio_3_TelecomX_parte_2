# 📡 TelecomX — Parte 2: Predicción de Cancelación de Clientes (Churn)

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3+-orange?logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-2.0+-150458?logo=pandas&logoColor=white)
![Jupyter](https://img.shields.io/badge/Notebook-Jupyter-F37626?logo=jupyter&logoColor=white)
![Status](https://img.shields.io/badge/Status-Completo-brightgreen)

---

## 📌 Propósito del Análisis

Este proyecto corresponde a la **segunda etapa del Challenge TelecomX** de Alura LATAM. El objetivo principal es **predecir el churn (cancelación) de clientes** de una empresa de telecomunicaciones, utilizando variables relevantes del perfil del cliente, su tipo de contrato y sus patrones de consumo.

La empresa TelecomX enfrenta una tasa de cancelación del **25.72%**, lo que representa una pérdida significativa de ingresos. A través del modelado predictivo con Machine Learning, buscamos:

- Identificar **qué clientes tienen mayor riesgo de cancelar** antes de que lo hagan
- Determinar **qué factores influyen más** en la decisión de cancelación
- Proveer **insights accionables** para que el equipo de retención pueda intervenir de forma proactiva y personalizada

> Este proyecto es la continuación directa de la **Parte 1 (EDA)**, donde se realizó el análisis exploratorio y la limpieza inicial de los datos. Se recomienda revisar `TelecomX_LATAM.ipynb` primero para obtener el contexto completo.

---

## 📁 Estructura del Proyecto

```
telecomx-churn/
│
├── 📓 TelecomX_LATAM.ipynb           # Parte 1: EDA y limpieza de datos
├── 📓 TelecomX_Parte2_ML.ipynb       # Parte 2: Modelado predictivo (ML) ← principal
│
├── 📊 visualizaciones/
│   ├── 01_proporcion_churn.png        # Distribución y desbalance de clases
│   ├── 02_correlacion_heatmap.png     # Heatmap de correlación de variables
│   ├── 03_tenure_vs_churn.png         # Análisis Tenure × Cancelación
│   ├── 04_gasto_vs_churn.png          # Análisis Gasto × Cancelación
│   ├── 05_evaluacion_modelos.png      # Matrices de confusión + curvas ROC
│   ├── 06_coeficientes_logistica.png  # Importancia de variables (Reg. Logística)
│   ├── 07_importancia_rf.png          # Importancia de variables (Random Forest)
│   └── 08_comparacion_importancia.png # Comparación entre modelos
│
└── README.md                          # Este archivo
```

> 💡 Los gráficos se generan automáticamente al ejecutar el notebook y quedan guardados en el directorio de trabajo.

---

## 🗂️ Dataset

| Atributo | Detalle |
|----------|---------|
| **Fuente** | [TelecomX Data — Alura LATAM GitHub](https://raw.githubusercontent.com/ingridcristh/challenge2-data-science-LATAM/main/TelecomX_Data.json) |
| **Formato original** | JSON con columnas anidadas |
| **Clientes** | 7,267 |
| **Variables (post-procesamiento)** | 29 |
| **Variable objetivo** | `Churn` (1 = canceló, 0 = activo) |
| **Churn rate** | 25.72% |

Los datos se cargan directamente desde la URL en el notebook, sin necesidad de descarga previa.

---

## 🔢 Clasificación de Variables

### Variables Numéricas (continuas)

| Variable | Descripción |
|----------|-------------|
| `tenure` | Meses de permanencia del cliente |
| `MonthlyCharges` | Cargo mensual en dólares |
| `TotalCharges` | Cargo total acumulado en dólares |

Estas variables fueron **estandarizadas con StandardScaler** para el modelo de Regresión Logística (media = 0, desviación estándar = 1), ya que este algoritmo es sensible a la escala. Para Random Forest no se aplicó normalización, dado que los árboles de decisión no dependen de la magnitud de los datos.

### Variables Categóricas Binarias (Yes/No → 1/0)

`Partner` · `Dependents` · `PhoneService` · `MultipleLines` · `OnlineSecurity` · `OnlineBackup` · `DeviceProtection` · `TechSupport` · `StreamingTV` · `StreamingMovies` · `PaperlessBilling`

También: `gender` → Male = 1, Female = 0

### Variables Categóricas con Múltiples Categorías (One-Hot Encoding)

| Variable | Categorías |
|----------|-----------|
| `InternetService` | DSL / Fiber optic / No |
| `Contract` | Month-to-month / One year / Two year |
| `PaymentMethod` | Electronic check / Mailed check / Bank transfer / Credit card |

Se aplicó **One-Hot Encoding** creando una columna binaria por cada categoría. No se usó `drop_first` para mantener transparencia en la interpretación de los coeficientes.

---

## 🛠️ Proceso de Preparación de Datos

### 1. Carga y desempaquetado

El JSON original tiene columnas anidadas (`customer`, `phone`, `internet`, `account`). Se utilizó `pd.json_normalize()` para expandir cada columna en sus campos individuales y consolidar todo en un único DataFrame.

### 2. Limpieza

- `TotalCharges` convertida a numérico (contenía espacios en blanco para clientes nuevos → reemplazados por 0)
- `Churn` mapeada de Yes/No a 1/0
- Eliminación de duplicados y filas con `Churn` nulo

### 3. Eliminación de columnas irrelevantes

Se eliminó `customerID` por ser un identificador único que no aporta información predictiva y puede introducir ruido en los modelos.

### 4. Verificación del desbalance de clases

```
No Churn (0):  5,394 clientes  →  74.28%
Churn    (1):  1,869 clientes  →  25.72%
Ratio de desbalance: 2.89 : 1
```

Se detectó un dataset desbalanceado. Para corregirlo se aplicó **Oversampling (Random Oversampling)** sobre la clase minoritaria (Churn = 1), duplicando muestras aleatorias hasta igualar la clase mayoritaria.

> ⚠️ El balanceo se aplica **exclusivamente al set de entrenamiento**, para que el set de prueba conserve la distribución real del negocio y las métricas sean representativas del mundo real.

### 5. División Train / Test

```python
train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
```

| Conjunto | Tamaño | Proporción |
|----------|--------|-----------|
| Entrenamiento | ~5,813 muestras | 80% |
| Prueba | ~1,454 muestras | 20% |

Se utilizó `stratify=y` para garantizar que ambos conjuntos mantengan la misma proporción de clases que el dataset original.

### 6. Justificaciones de las decisiones de modelado

| Decisión | Justificación |
|----------|--------------|
| Normalizar solo para Regresión Logística | Los modelos lineales calculan coeficientes sobre la magnitud de los datos. Sin escalar, variables como `TotalCharges` (~$2,000) dominarían sobre `SeniorCitizen` (0 o 1) |
| No normalizar para Random Forest | Los árboles dividen por umbrales relativos, no por distancias; la escala no afecta el resultado |
| Oversampling en lugar de undersampling | Con ~7,000 filas, el undersampling reduciría demasiado los datos de entrenamiento, perdiendo información valiosa |
| Balanceo solo en train | Aplicar balanceo al test contaminaría las métricas, haciendo que no reflejen el desempeño real del modelo |
| `max_depth=12` en Random Forest | Limita la complejidad para reducir overfitting sin sacrificar capacidad predictiva |
| `C=1.0` en Regresión Logística | Regularización L2 estándar que controla el sobreajuste sin restringir demasiado los coeficientes |

---

## 📊 Gráficos e Insights del Análisis Exploratorio

### 1. Distribución del Churn — Desbalance de Clases

> *Archivo: `01_proporcion_churn.png`*

El 74.28% de los clientes permanece activo y solo el 25.72% cancela. Este desbalance es suficiente para sesgar los modelos hacia la clase mayoritaria si no se trata, generando alta accuracy pero bajo Recall sobre los churners.

**💡 Insight:** Sin tratamiento del desbalance, el modelo aprende a predecir siempre "No Churn" y obtiene 74% de accuracy sin detectar casi ningún cliente que cancela. El Oversampling es esencial para que aprenda los patrones reales de cancelación.

---

### 2. Correlación de Variables con Churn

> *Archivo: `02_correlacion_heatmap.png`*

Variables con mayor correlación positiva (mayor riesgo de cancelación):

| Variable | Correlación |
|----------|:-----------:|
| `Contract_Month-to-month` | +0.40 |
| `Payment_Electronic check` | +0.30 |
| `Internet_Fiber optic` | +0.31 |
| `MonthlyCharges` | +0.19 |

Variables con mayor correlación negativa (mayor retención):

| Variable | Correlación |
|----------|:-----------:|
| `tenure` | −0.35 |
| `Contract_Two year` | −0.30 |
| `TotalCharges` | −0.20 |

**💡 Insight:** El tipo de contrato y el tiempo de permanencia son los predictores más fuertes. Los clientes sin compromiso contractual y con poco tiempo en la empresa son el grupo de mayor riesgo.

---

### 3. Tenure × Churn — Tiempo de Contrato vs Cancelación

> *Archivo: `03_tenure_vs_churn.png`*

```
Rango de Tenure     Churn Rate    Clientes
────────────────────────────────────────────
0 - 6 meses          51.41% ⚠️     1,525
6 - 12 meses         34.71%           729
12 - 24 meses        28.13%         1,045
24 - 36 meses        20.86%           863
36 - 48 meses        18.47%           785
48 - 72 meses         9.22% ✅       2,309
```

- Clientes **con Churn**: tenure promedio = **17.98 meses** (mediana: 10 meses)
- Clientes **sin Churn**: tenure promedio = **37.32 meses** (mediana: 37 meses)
- Diferencia: **19.34 meses** menos en promedio

**💡 Insight:** Los primeros 6 meses son el período crítico con más del 51% de cancelaciones. Un cliente que supera el primer año tiene una probabilidad de churn significativamente menor. Esto señala la necesidad urgente de un programa de onboarding y retención temprana.

---

### 4. Gasto × Churn — Cargos vs Cancelación

> *Archivo: `04_gasto_vs_churn.png`*

| Métrica | Clientes con Churn | Clientes sin Churn |
|---------|:-----------------:|:-----------------:|
| Monthly Charges promedio | ~$74.44 | ~$61.27 |
| Total Charges promedio | ~$1,531 | ~$2,555 |

El scatter plot Tenure vs MonthlyCharges revela un patrón claro: los churners se concentran en la zona de **bajo tenure + altos cargos mensuales**, mientras que los clientes leales tienen mayor tenure con una distribución de cargos más variada.

**💡 Insight:** No es el gasto total lo que impulsa el churn, sino la relación entre el precio percibido y el valor recibido durante los primeros meses. Un cliente nuevo que paga mucho y aún no percibe el valor del servicio tiene alto riesgo de abandonar.

---

## 🤖 Modelos y Resultados

### Comparación de Métricas

| Métrica | Regresión Logística | Random Forest |
|---------|:-------------------:|:-------------:|
| Accuracy (train) | ~0.76 | ~0.92 |
| Accuracy (test) | ~0.75 | ~0.80 |
| Precision | ~0.49 | ~0.58 |
| Recall | ~0.80 | ~0.73 |
| F1-Score | ~0.61 | ~0.64 |
| ROC-AUC | ~0.84 | ~0.86 |

**Random Forest** obtiene mejor ROC-AUC y F1-Score → recomendado para scoring general.

**Regresión Logística** obtiene mayor Recall → más adecuada cuando el objetivo es capturar el máximo de churners, dado que en churn el costo de no detectar a un cliente que se va (falso negativo) es mayor que el de alertar erróneamente a uno que se queda (falso positivo).

### Variables Más Importantes

Ambos modelos coinciden en el mismo conjunto de variables clave:

1. `tenure` — el tiempo de contrato es el predictor más fuerte
2. `MonthlyCharges` — cargos mensuales altos correlacionan con cancelación
3. `TotalCharges` — inversamente relacionado con el churn
4. `Contract_Month-to-month` — mayor riesgo que contratos anuales
5. `Payment_Electronic check` — método de pago más asociado al churn

---

## ▶️ Instrucciones de Ejecución

### Requisitos

Python 3.10 o superior. Instalá las dependencias con:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn requests
```

O si usás un entorno virtual:

```bash
python -m venv venv
source venv/bin/activate      # Linux / Mac
venv\Scripts\activate         # Windows

pip install pandas numpy matplotlib seaborn scikit-learn requests
```

### En Google Colab (recomendado)

1. Abrí [Google Colab](https://colab.research.google.com/)
2. Subí el archivo `TelecomX_Parte2_ML.ipynb`
3. Ejecutá todas las celdas en orden con `Runtime → Run all`

> Las librerías necesarias ya vienen preinstaladas en Colab. No se necesita instalar nada adicional.

### En Jupyter Notebook local

```bash
git clone https://github.com/javito72/telecomx-churn.git
cd telecomx-churn
jupyter notebook TelecomX_Parte2_ML.ipynb
```

### Carga de datos

Los datos se cargan automáticamente al ejecutar la primera celda del notebook. No es necesario descargar ni configurar ningún archivo:

```python
url = 'https://raw.githubusercontent.com/ingridcristh/challenge2-data-science-LATAM/main/TelecomX_Data.json'
response = requests.get(url)
df_raw = pd.DataFrame(response.json())
```

> ⚠️ Se requiere conexión a internet para la carga inicial de los datos.

### Orden de ejecución recomendado

```
1. TelecomX_LATAM.ipynb         → EDA y análisis exploratorio (contexto)
2. TelecomX_Parte2_ML.ipynb     → Modelado predictivo (resultado final)
```

---

## 🛠️ Tecnologías Utilizadas

| Librería | Versión recomendada | Uso principal |
|----------|:------------------:|---------------|
| `pandas` | 2.0+ | Manipulación y limpieza de datos |
| `numpy` | 1.24+ | Operaciones numéricas |
| `matplotlib` | 3.7+ | Visualización de datos |
| `seaborn` | 0.12+ | Gráficos estadísticos |
| `scikit-learn` | 1.3+ | Modelos ML, preprocesamiento y evaluación |
| `requests` | 2.28+ | Carga de datos desde URL |

---

## 👤 Autor: Christian Javier Lemos

Desarrollado como parte del **Challenge de Ciencia de Datos — Alura LATAM**

---

*"Los datos no mienten: retener un cliente siempre es más barato que adquirir uno nuevo."*

*"Los datos no mienten: retener un cliente es siempre más barato que adquirir uno nuevo."*

