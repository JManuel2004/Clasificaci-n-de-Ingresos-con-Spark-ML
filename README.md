# Clasificación de ingresos con Spark ML

Pipeline local que estima si el ingreso de una persona supera los 50 000 dólares. El conjunto de entrenamiento es sintético y la etiqueta sale de una regla fija, así que el mismo comando reproduce el mismo archivo sin descargas externas.

Las métricas de holdout miden qué tan bien el modelo recupera esa regla. Describen este generador, no el rendimiento sobre el censo Adult de UCI.

## Requisitos

- Python 3.9 o superior
- JDK 17 o superior, con `JAVA_HOME` apuntando al JDK. [Eclipse Temurin 17](https://adoptium.net/) sirve.
- En Windows el primer arranque de Spark compila un `winutils.exe` mínimo y un sistema de archivos local. Hace falta el compilador `csc.exe` de .NET Framework 4, que viene con Windows, y `javac`, que viene con el JDK. Quien ya tenga un Hadoop con `bin\winutils.exe` puede exportar `HADOOP_HOME` y se reutiliza ese binario.

## Instalación

```bash
python -m venv .venv
```

Windows:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```

macOS y Linux:

```bash
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

## Uso

Desde la raíz del repositorio:

```bash
income-generate
income-train
income-score
```

Los mismos pasos sin instalar los scripts:

```bash
python -m income_pipeline generate
python -m income_pipeline train
python -m income_pipeline score
```

`income-generate` escribe 20 000 filas en `data/raw/adult_income_sample.csv` con la semilla 42. `income-train` ajusta una regresión logística sobre un holdout del 20 % y guarda el modelo en `models/income_lr`. `income-score` puntúa `data/samples/applicants.csv` y escribe `data/scored/applicants.csv`.

Salidas de entrenamiento:

| Ruta | Contenido |
| --- | --- |
| `models/income_lr` | Pipeline de Spark ML ajustado |
| `artifacts/metrics.json` | AUC, precisión, recall y F1 en train y en test |
| `artifacts/quality_report.json` | Filas de entrada, rechazos y duplicados |
| `data/scored/applicants.csv` | Etiqueta predicha y probabilidad de `>50K` |

Esas rutas están en `.gitignore`. El CSV de entrenamiento también: se vuelve a crear con `income-generate`.

## Contrato de los datos

| Columna | Uso |
| --- | --- |
| `age` | Edad, de 16 a 100 |
| `sex` | `Female` o `Male` |
| `workclass` | `Gov`, `Private` o `Self-emp` |
| `fnlwgt` | Peso muestral del censo. Se conserva en el archivo y se queda fuera del modelo |
| `education` | Nivel educativo, codificado con un orden fijo |
| `hours_per_week` | Horas, de 1 a 99 |
| `label` | `>50K` o `<=50K`. El archivo a puntuar no la trae |

`fnlwgt` describe el diseño de la muestra, no a la persona, por eso no entra al vector de features. La clase positiva es siempre `>50K` = 1, aunque sea la clase mayoritaria.

La probabilidad sintética de `>50K` empieza en 0.12 y suma 0.28 con título universitario, 0.12 adicional con maestría o doctorado, 0.16 entre 30 y 55 años, 0.12 con 40 horas o más, y 0.05 en el sector privado. El tope es 0.95. El código vive en `positive_probability`.

## Entrenamiento

1. El CSV se lee con un esquema explícito. Un valor que no se puede convertir queda nulo y la fila se rechaza.
2. Se descartan filas fuera del contrato y duplicados exactos. Si la fracción inválida supera el límite, o falta una de las dos clases, el entrenamiento se detiene y deja el reporte de calidad.
3. El holdout se hace antes de ajustar imputers, indexadores y el modelo. Los pesos de clase se calculan solo en el pliegue de entrenamiento.
4. Las features numéricas son edad, horas, el ordinal de educación y un indicador de 40 horas o más. Sexo y tipo de empleo se codifican con one-hot. Las numéricas se estandarizan aparte para no reescalar las columnas binarias.
5. El modelo es una regresión logística binomial con regularización L2 (`regParam` 0.01) y estandarización interna apagada.

## Estructura

```text
src/income_pipeline/     comandos generate, train y score
data/raw/                CSV de entrenamiento, generado
data/samples/            solicitantes sin etiqueta
tests/                   pruebas del contrato, el split y el modelo
.github/workflows/       pytest en Ubuntu y Windows
```

## Pruebas

```bash
pytest
```

GitHub Actions ejecuta la misma suite en Ubuntu y en Windows, con Java 17.

## Límites

El generador no reproduce la distribución del censo. Un AUC alto aquí significa que la regresión sigue la regla documentada. Para datos nuevos, las categorías y los rangos tienen que respetar el contrato; las filas que no lo hacen se escriben en `data/scored/applicants.rejected.csv`.
