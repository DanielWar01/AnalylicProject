import pandas as pd
from pandasgui import show

dfStudents = pd.read_csv('./StudentPerformanceFactors.csv')

#Estructura y el resumen de información del DataFrame
dfStudents.info()

#La clase del objeto o su tipo de dato categórico o numérico
type(dfStudents)

#Tipo de datos de cada columna
print(dfStudents.dtypes)

#Obtención de un resumen estadístico de cada columna numérica
print(dfStudents.describe())

#Mostrar las primeras 15 filas
print(dfStudents.head(15))

#Mostrar las últimas 15 filas
print(dfStudents.tail(15))