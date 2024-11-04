import pandas as pd
from pandasgui import show

dfStudents = pd.read_csv('./StudentPerformanceFactors.csv')

############### PREPROCESAMIENTO ####################
#Estructura y el resumen de información del DataFrame
print(f"\nInformación general del DataFrame")
dfStudents.info()

#La clase del objeto o su tipo de dato categórico o numérico
print(f"\nClase y tipo de dato de cada columna")
type(dfStudents)

#Tipo de datos de cada columna
print(f"\nTipo de dato por columna")
print(dfStudents.dtypes)

#Obtención de un resumen estadístico de cada columna numérica
print(f"\nResumen estadístico")
print(dfStudents.describe())

#Mostrar las primeras 15 filas
print(f"\nPrimeras 15 columnas")
print(dfStudents.head(15))

#Mostrar las últimas 15 filas
print(f"\nÚltimas 15 columnas")
print(dfStudents.tail(15))

#Identificación de valores nulos o faltantes
print(f"\nIdentificación de valores nulos")
print(dfStudents.isnull().sum())

#Tratamiento de valores faltantes o nulos
for column in dfStudents.columns:
    if dfStudents[column].dtype == 'object':  # Columna categórica
        dfStudents[column].fillna(dfStudents[column].mode()[0], inplace=True)
    else:  # Columna numérica
        dfStudents[column].fillna(dfStudents[column].mean(), inplace=True)

#Eliminación de valores duplicados
dfStudents = dfStudents.drop_duplicates()

show(dfStudents)

