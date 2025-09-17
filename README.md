# Clasificación de Ingresos con Spark ML

## Descripción del Proyecto

Sistema de clasificación binaria desarrollado para **DataPros** que predice si una persona gana más de $50K al año utilizando características demográficas y laborales. El modelo está implementado con **Apache Spark ML** y **Logistic Regression**.

##  Dataset

El archivo `adult_income_sample.csv` contiene 2000 registros simulados con las siguientes características:

| Columna | Descripción |
|---------|-------------|
| `age` | Edad de la persona (años) |
| `sex` | Género (Male, Female) |
| `workclass` | Tipo de empleo (Private, Self-emp, Gov) |
| `fnlwgt` | Peso estadístico asociado al registro |
| `education` | Nivel educativo (Bachelors, HS-grad, 11th, Masters, etc.) |
| `hours_per_week` | Horas trabajadas por semana |
| `label` | Clase objetivo: >50K o <=50K |

##  Estructura del Proyecto

```
C:\spark_ml_classification\
├── adult_income_sample.csv              # Datos de entrenamiento (2000 registros)
├── generate_data.py                     # Script para generar datos simulados
├── income_classification.py             # Script principal de PySpark
├── income_classification_notebook.ipynb # Jupyter notebook con evidencia
└── README.md                           # Este archivo
```

##  Instalación y Ejecución

### Requisitos Previos

1. **Python 3.7+**
2. **Java 8 o 11** (requerido por Spark)
3. **PySpark**

###  **Archivos del Proyecto**

| Archivo | Descripción |
|---------|-------------|
| `generate_data.py` | Generador del dataset CSV con 2000 registros simulados |
| `adult_income_sample.csv` | Dataset generado para entrenamiento del modelo |
| `income_classification_colab.ipynb` | Notebook principal para Google Colab |
| `README.md` | Documentación completa del proyecto |

### Instalación

```bash
# Para generar datos localmente (opcional)
pip install pandas numpy

# En Google Colab se instala automáticamente:
# pip install pyspark
```

### Generación del Dataset

```bash
# Ejecutar el generador de datos (solo si necesitas regenerar el CSV)
python generate_data.py
# Esto crea: adult_income_sample.csv con 2000 registros
```

##  Componentes Técnicos

### Pipeline de Machine Learning

1. **StringIndexer**: Convierte variables categóricas a índices numéricos
2. **OneHotEncoder**: Transforma índices a vectores binarios
3. **VectorAssembler**: Combina todas las características en un vector
4. **LogisticRegression**: Modelo de clasificación binaria

### Características del Modelo

- **Algoritmo**: Regresión Logística
- **Máximo de iteraciones**: 100
- **Parámetro de regularización**: 0.01
- **Variables predictoras**: 
  - Numéricas: age, fnlwgt, hours_per_week
  - Categóricas: sex, workclass, education (codificadas)

##  Resultados Esperados

El modelo genera:

1. **Predicciones binarias**: 0 (<=50K) o 1 (>50K)
2. **Probabilidades**: Vector con probabilidades para cada clase
3. **Métricas de evaluación**:
   - Precisión (Accuracy)
   - Área bajo la curva ROC (AUC)
   - Matriz de confusión

### Ejemplo de Salida

```
 Métricas de evaluación:
   Área bajo la curva ROC (AUC): 0.8234
   Precisión (Accuracy): 0.7650
   Predicciones correctas: 1530
   Total de predicciones: 2000
```

##  Casos de Uso

### Predicción con Nuevos Datos

El notebook incluye 9 casos de prueba que demuestran diferentes perfiles:

1. **Profesional con título**: 35 años, Bachelors, 45h/semana → Probable >50K
2. **Joven con educación básica**: 25 años, HS-grad, 35h/semana → Probable <=50K
3. **Empleado gobierno con maestría**: 45 años, Masters, 50h/semana → Probable >50K
4. **Trabajador independiente joven**: 22 años, 11th grade, 20h/semana → Probable <=50K
5. **Profesional senior con doctorado**: 55 años, Doctorate, 60h/semana → Muy probable >50K

##  Análisis y Reflexiones

### Factores Influyentes

1. **Educación**: Mayor nivel educativo correlaciona con ingresos altos
2. **Edad**: Experiencia laboral (edad media) favorece ingresos altos
3. **Horas trabajadas**: Más de 40 horas/semana aumenta probabilidad >50K
4. **Tipo de empleo**: Sector privado muestra mejor performance

### Limitaciones

- Datos simulados (no representan población real)
- Variables limitadas (podrían incluirse más características)
- No hay división train/test (overfitting potencial)

##  Extensiones Posibles

1. **División train/test**: Validación más robusta
2. **Más algoritmos**: Random Forest, Gradient Boosting
3. **Feature engineering**: Nuevas variables derivadas
4. **Validación cruzada**: Mejor estimación de rendimiento
5. **Datos reales**: Usar dataset Adult Income original

## Contacto

Proyecto desarrollado para **DataPros**  
Sistema de predicción de ingresos con Apache Spark ML

---

**¡El modelo está listo para predecir ingresos basado en características demográficas y laborales!** 

---

##  **Ejecución en Google Colab**

###  **Archivo para Colab:** `income_classification_colab.ipynb`

Este proyecto incluye una versión **completamente adaptada para Google Colab** que permite ejecutar el mismo pipeline sin problemas de configuración local.

###  **Instrucciones para ejecutar en Google Colab:**

#### **Paso 1: Preparar el dataset**
```bash
# 1. Ejecuta localmente el generador de datos (si aún no lo hiciste):
python generate_data.py

# 2. Esto crea: adult_income_sample.csv (2000 registros)
```

#### **Paso 2: Subir a Google Drive**
1.  Abre [Google Drive](https://drive.google.com)
2.  Sube el archivo `adult_income_sample.csv` a la raíz de tu Drive
3.  Verifica que esté en: `/content/drive/MyDrive/adult_income_sample.csv`

#### **Paso 3: Ejecutar en Colab**
1.  Abre [Google Colab](https://colab.research.google.com/)
2.  Sube el archivo `income_classification_colab.ipynb`
3.  Ejecuta **todas las celdas secuencialmente**
4.  Autoriza el acceso a Google Drive cuando se solicite

###  **Ventajas de la versión Colab:**

####  **Configuración automática:**
- PySpark se instala automáticamente
- No requiere configuración de Java local
- Entorno limpio y consistente

####  **Integración con Google Drive:**
- Lee el mismo CSV usado localmente
- Mantiene compatibilidad entre entornos
- Facilita colaboración del equipo

####  **Resultados idénticos:**
- Mismo pipeline de ML (StringIndexer + OneHotEncoder + VectorAssembler + LogisticRegression)
- Mismas métricas de evaluación
- Predicciones consistentes

####  **Evidencia de ejecución:**
- Salida detallada de cada etapa
- Métricas de rendimiento visibles
- Predicciones interpretadas
- Fácil generación de PDF


---

##  **Evidencia de Ejecución**

###  **Resultados obtenidos:**
- Dataset de 2000 registros procesado exitosamente
- Pipeline completo implementado con 10 etapas
- Modelo entrenado con métricas de evaluación (AUC y Accuracy)
- Predicciones realizadas sobre 9 casos nuevos

###  **Capturas recomendadas:**
1. Notebook ejecutándose con datos cargados
2. Salida del entrenamiento del modelo
3. Métricas de evaluación del modelo
4. Predicciones sobre nuevos datos
5. Resumen final del proyecto


