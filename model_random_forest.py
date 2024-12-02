import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

# Cargar el dataset
df = pd.read_csv("./StudentPerformanceFactorsClean.csv")

################# MODELO RANDOM FOREST REGRESSOR #####################

########### Selección de variables ###########
x_1 = "Hours_Studied"
x_2 = "Attendance"
y = "Exam_Score"

############### Creación de resultados #################
#************** Establecimiento de variables ***********
variables_x = df[[x_1, x_2]]
variable_y = df[y]

# Dividir los datos en entrenamiento y prueba
X_train, X_test, Y_train, Y_test = train_test_split(
    variables_x,
    variable_y,
    test_size=0.2,
    random_state=1
)

# Escalado de las características
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

############### Creación del modelo ###############
forest = RandomForestRegressor(
    n_estimators=300,
    max_features='sqrt',
    max_depth=30,
    random_state=1
)

# Ajustar el modelo con los datos de entrenamiento
forest.fit(X_train, Y_train)

# Predicciones
Y_train_pred = forest.predict(X_train)
Y_test_pred = forest.predict(X_test)

############### Evaluación del modelo ###############
# Métricas de desempeño
train_r2 = r2_score(Y_train, Y_train_pred)
test_r2 = r2_score(Y_test, Y_test_pred)
test_rmse = np.sqrt(mean_squared_error(Y_test, Y_test_pred))

print(f"R² en entrenamiento: {train_r2:.2f}")
print(f"R² en prueba: {test_r2:.2f}")
print(f"RMSE en prueba: {test_rmse:.2f}")
print("\n***********Predicciones con el modelo************\n")
# Predicción con los datos de prueba
Y_test_pred = forest.predict(X_test)

# Mostrar algunas predicciones junto con los valores reales
predicciones = pd.DataFrame({
    'Real': Y_test.values,
    'Predicción': Y_test_pred
})
print(predicciones.head(10))  # Mostrar las primeras 10 filas

# Graficar valores reales vs predicciones
plt.scatter(Y_test, Y_test_pred, alpha=0.7)
plt.xlabel("Valores reales")
plt.ylabel("Predicciones")
plt.title("Comparación de valores reales vs predicciones")
plt.plot([Y_test.min(), Y_test.max()], [Y_test.min(), Y_test.max()], color='red', linestyle='--')
plt.show()