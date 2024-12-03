import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.metrics import r2_score
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

df = pd.read_csv("./StudentPerformanceFactorsClean.csv")

######### REGRESIÓN LINEAL MÚLTIPLE ##########

########### Selección de variables ###########

x_1 = "Hours_Studied"
x_2 = "Attendance"
y = "Exam_Score"

############### Creación de resultados #################
#************** Establecimiento de variables ***********

variables_x = [x_1, x_2]
variable_y = y

#**************** Generación del modelo ****************

modelo = LinearRegression()

#************** Entrenamiento del modelo ***************

modelo.fit(df[variables_x], df[variable_y])

#****** Obtención de coeficientes de las variables ******
#*********** independientes y la intercepción ***********

print('Coeficientes: ', modelo.coef_)
print('Intercepción: ', modelo.intercept_)

#************ Imprimir la ecuación del plano ************

print(f'Ecuación del plano: Puntaje del examen = '
      f'{round(modelo.coef_[0], 3)} * Horas de estudio + '
      f'{round(modelo.coef_[1], 3)} * Asistencia + '
      f'{round(modelo.intercept_, 3)}')
print(f'Coeficiente de determinación: '
      f'{round(r2_score(df[variable_y], modelo.predict(df[variables_x])), 3)}')

# Predecir un puntaje de examen para 6 horas de estudio y 85% de asistencia
nuevo_dato = [[5.594405594405594, 98]]
prediccion = modelo.predict(nuevo_dato)
print(f"La predicción para {nuevo_dato[0][0]} horas de estudio y un {nuevo_dato[0][1]}% de asistencia. \nPredicción: {prediccion[0]:.2f}\nValor real: 74")


#******************* Gráfica en 3D ***********************

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(df[x_1], df[x_2], df[y], color='red')


#************ Crear un meshgrid para el plano ************

x = np.linspace(df[x_1].min(), df[x_1].max(), num=10)
y = np.linspace(df[x_2].min(), df[x_2].max(), num=10)
x, y = np.meshgrid(x, y)

#**** Calcular los valores en z (Puntaje del examen)  *****

z = modelo.intercept_ + modelo.coef_[0] * x + modelo.coef_[1] * y

#****************** Graficar el plano *********************

ax.plot_surface(x, y, z, alpha=0.5)
ax.set_xlabel(x_1)
ax.set_ylabel(x_2)
ax.set_zlabel("Exam_score")
plt.show()
