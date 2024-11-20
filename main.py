import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def preprocess_data(file_path):
    """
    Preprocesa los datos del archivo CSV y maneja los valores nulos correctamente.
    """
    # Lectura del dataset
    df = pd.read_csv(file_path)

    # Numero de valores faltantes o anomalas
    missing_values = df.isna().sum()
    print("\nValores faltantes por columna:")
    print(missing_values)

    # Crear una copia explícita del DataFrame para no alterar el original
    df = df.copy()

    # Tratamiento de valores faltantes o nulos
    for column in df.columns:
        if df[column].dtype == 'object':  # Columna categórica
            df[column] = df[column].fillna(df[column].mode()[0])
        else:  # Columna numérica
            df[column] = df[column].fillna(df[column].mean())

    # Eliminación de valores duplicados
    df = df.drop_duplicates()

    return df


def crear_visualizaciones(df):
    """
    Crea visualizaciones para análisis exploratorio de datos
    """
    # Configuración del backend de matplotlib
    plt.switch_backend('Agg')  # Usar el backend Agg que no requiere GUI

    # Configuración del estilo
    sns.set_theme(style="whitegrid")

    # Variables numéricas y categóricas
    numerical_cols = ['Hours_Studied', 'Attendance', 'Sleep_Hours',
                      'Previous_Scores', 'Physical_Activity', 'Exam_Score']
    categorical_cols = ['Parental_Involvement', 'Access_to_Resources',
                        'Extracurricular_Activities', 'Motivation_Level',
                        'Family_Income', 'School_Type', 'Gender']

    # Lista para almacenar todas las figuras
    figs = []

    # 1. Histogramas para variables numéricas
    fig1, axes = plt.subplots(3, 2, figsize=(15, 20))
    fig1.suptitle('Distribución de Variables Numéricas', fontsize=16)
    for idx, col in enumerate(numerical_cols):
        row = idx // 2
        col_idx = idx % 2
        sns.histplot(data=df, x=col, kde=True, ax=axes[row, col_idx])
        axes[row, col_idx].set_title(f'Distribución de {col}')
    plt.tight_layout()
    figs.append(fig1)

    # 2. Boxplots para variables numéricas
    fig2 = plt.figure(figsize=(15, 8))
    sns.boxplot(data=df[numerical_cols])
    plt.xticks(rotation=45)
    plt.title('Boxplots de Variables Numéricas')
    plt.tight_layout()
    figs.append(fig2)

    # 3. Matriz de correlación
    fig3 = plt.figure(figsize=(10, 8))
    correlation_matrix = df[numerical_cols].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title('Matriz de Correlación')
    plt.tight_layout()
    figs.append(fig3)

    # 4. Gráficos de barras para variables categóricas
    fig4, axes = plt.subplots(3, 3, figsize=(20, 15))
    fig4.suptitle('Distribución de Variables Categóricas', fontsize=16)
    for idx, col in enumerate(categorical_cols):
        row = idx // 3
        col_idx = idx % 3
        sns.countplot(data=df, x=col, ax=axes[row, col_idx])
        axes[row, col_idx].tick_params(axis='x', rotation=45)

    # Eliminar subplots vacíos si los hay
    for i in range(len(categorical_cols), 9):
        row = i // 3
        col_idx = i % 3
        fig4.delaxes(axes[row, col_idx])

    plt.tight_layout()
    figs.append(fig4)

    # 5. Scatter plots vs Exam_Score
    fig5, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig5.suptitle('Relaciones con Exam_Score', fontsize=16)
    axes = axes.ravel()
    for idx, col in enumerate(numerical_cols[:-1]):  # Excluimos Exam_Score
        sns.scatterplot(data=df, x=col, y='Exam_Score', ax=axes[idx])
        axes[idx].set_title(f'{col} vs Exam_Score')

    # Eliminar subplot vacío si lo hay
    if len(numerical_cols[:-1]) < 6:
        for i in range(len(numerical_cols[:-1]), 6):
            fig5.delaxes(axes[i])

    plt.tight_layout()
    figs.append(fig5)

    # 6. Violin plots para variables categóricas vs Exam_Score
    fig6, axes = plt.subplots(3, 3, figsize=(20, 15))
    fig6.suptitle('Distribución de Exam_Score por Categorías', fontsize=16)
    axes = axes.ravel()
    for idx, col in enumerate(categorical_cols):
        sns.violinplot(data=df, x=col, y='Exam_Score', ax=axes[idx])
        axes[idx].tick_params(axis='x', rotation=45)
        axes[idx].set_title(f'Exam_Score por {col}')

    # Eliminar subplots vacíos si los hay
    for i in range(len(categorical_cols), 9):
        fig6.delaxes(axes[i])

    plt.tight_layout()
    figs.append(fig6)

    # 7. Pair plot para variables numéricas principales
    fig7 = sns.pairplot(df[['Hours_Studied', 'Attendance', 'Previous_Scores', 'Exam_Score']],
                        diag_kind='kde')
    figs.append(fig7.fig)

    # 8. Box plots de Exam_Score por cada variable categórica
    fig8, axes = plt.subplots(3, 3, figsize=(20, 15))
    fig8.suptitle('Distribución de Exam_Score por Variables Categóricas', fontsize=16)
    axes = axes.ravel()
    for idx, col in enumerate(categorical_cols):
        sns.boxplot(data=df, x=col, y='Exam_Score', ax=axes[idx])
        axes[idx].tick_params(axis='x', rotation=45)
        axes[idx].set_title(f'Exam_Score por {col}')

    # Eliminar subplots vacíos si los hay
    for i in range(len(categorical_cols), 9):
        fig8.delaxes(axes[i])

    plt.tight_layout()
    figs.append(fig8)

    return figs


def main():
    # Cargar y preprocesar datos
    df = preprocess_data('./StudentPerformanceFactors.csv')

    # Mostrar información del DataFrame
    print("\nInformación general del DataFrame")
    df.info()

    print("\nTipo de dato por columna")
    print(df.dtypes)

    print("\nResumen estadístico por columna:")
    for column in df.columns:
        print(f"\n--- Descripción de {column} ---")
        if df[column].dtype in ['int64', 'float64']:
            # Para columnas numéricas, usar describe() completo
            print(df[column].describe())
        else:
            # Para columnas categóricas, mostrar frecuencia de valores
            print(df[column].value_counts())
            print("\nFrecuencia porcentual:")
            print(df[column].value_counts(normalize=True) * 100)



    # Crear visualizaciones
    figs = crear_visualizaciones(df)

    # Guardar las visualizaciones
    for i, fig in enumerate(figs):
        fig.savefig(f'visualization_{i}.png')
        plt.close(fig)


if __name__ == "__main__":
    main()