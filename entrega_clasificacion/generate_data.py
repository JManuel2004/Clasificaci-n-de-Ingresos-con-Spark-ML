import pandas as pd
import numpy as np
import random

# Configurar semilla para reproducibilidad
np.random.seed(42)
random.seed(42)

# Definir valores posibles para variables categóricas
sex_values = ['Male', 'Female']
workclass_values = ['Private', 'Self-emp', 'Gov']
education_values = ['Bachelors', 'HS-grad', '11th', 'Masters', 'Some-college', 
                   'Assoc-voc', 'Doctorate', '9th', '10th', '12th', 'Preschool']

# Generar datos simulados
data = []
for i in range(2000):
    age = np.random.randint(18, 75)
    sex = random.choice(sex_values)
    workclass = random.choice(workclass_values)
    fnlwgt = np.random.randint(12000, 1500000)
    education = random.choice(education_values)
    hours_per_week = np.random.randint(1, 80)
    
    # Lógica simple para determinar la etiqueta basada en algunas características
    # Para hacer más realista, personas con mayor educación, edad media y más horas
    # tienen mayor probabilidad de ganar >50K
    prob_high_income = 0.2  # probabilidad base
    
    if education in ['Bachelors', 'Masters', 'Doctorate']:
        prob_high_income += 0.3
    if age >= 30 and age <= 55:
        prob_high_income += 0.2
    if hours_per_week >= 40:
        prob_high_income += 0.15
    if workclass == 'Private':
        prob_high_income += 0.1
    
    label = '>50K' if random.random() < prob_high_income else '<=50K'
    
    data.append([age, sex, workclass, fnlwgt, education, hours_per_week, label])

# Crear DataFrame
df = pd.DataFrame(data, columns=['age', 'sex', 'workclass', 'fnlwgt', 'education', 'hours_per_week', 'label'])

# Guardar como CSV
df.to_csv('C:/spark_ml_classification/adult_income_sample.csv', index=False)

print(f"Archivo generado con {len(df)} registros")
print("\nPrimeros 10 registros:")
print(df.head(10))
print("\nDistribución de etiquetas:")
print(df['label'].value_counts())
print("\nInfo del DataFrame:")
print(df.info())