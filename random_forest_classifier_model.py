import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Cargar el dataset
# Asegúrate de tener el dataset como un archivo CSV o cargado en un DataFrame
df = pd.read_csv('./StudentPerformanceFactorsClean.csv')

# Variables seleccionadas
selected_features = ['Family_Income', 'Teacher_Quality', 'Peer_Influence', 'Physical_Activity']
target_variable = 'Motivation_Level'

# Codificación de variables categóricas para ser transformadas a numéricas
label_encoders = {}
for col in selected_features + [target_variable]:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# División de datos en entrenamiento y prueba
X = df[selected_features]
y = df[target_variable]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modelo: Random Forest Classifier
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)


# Predicciones
y_pred = model.predict(X_test)

# Evaluación del modelo
print("Matriz de confusión:\n", confusion_matrix(y_test, y_pred))
print("\nReporte de clasificación:\n", classification_report(y_test, y_pred))

# Importancia de las variables
importances = model.feature_importances_
feature_importance = pd.DataFrame({'Variable': X.columns, 'Importancia': importances})
print(feature_importance.sort_values(by='Importance', ascending=False))
