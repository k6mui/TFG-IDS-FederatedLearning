import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler

# Crea una instancia del codificador de etiquetas y MinMaxScaler
label_encoder = LabelEncoder()
scaler = MinMaxScaler()

# Características para eliminar
delete_features = ['IPV4_SRC_ADDR', 'L4_SRC_PORT', 'IPV4_DST_ADDR', 'L4_DST_PORT', 'Dataset']

# Rutas de los archivos CSV del escenario heterogéneo
# paths = [
#     "../datasets/sampled/cicids_sampled.csv",
#     "../datasets/sampled/nb15_sampled.csv",
#     "../datasets/sampled/toniot_sampled.csv"
# ]

# Rutas de los archivos CSV del escenario no heterogéneo
paths = [
    "../datasets/balanced/first_balanced.csv",
    "../datasets/balanced/second_balanced.csv",
    "../datasets/balanced/third_balanced.csv"
]


# Nombres de los conjuntos de datos para la nueva columna 'dataset'
dataset_names = ['CIC', 'NB15', 'TONIOT']

# Número de filas para muestrear de cada DataFrame
numero_filas_muestreo = 100000

# Lista para almacenar los dataframes muestreados
sampled_dfs = []

# Cargar los archivos CSV, muestrear filas, eliminar características innecesarias y añadir la columna 'dataset'
for path, dataset_name in zip(paths, dataset_names):
    df = pd.read_csv(path, low_memory=False)
    df_sampled = df.sample(n=numero_filas_muestreo, random_state=82)
    df_sampled = df_sampled.drop(columns=delete_features)
    df_sampled['dataset'] = dataset_name
    sampled_dfs.append(df_sampled)

# Combina los tres dataframes muestreados en uno solo
df_total = pd.concat(sampled_dfs)

# Codificar la columna 'Attack' y luego eliminarla
df_total['Attack_encoded'] = label_encoder.fit_transform(df_total['Attack'])
df_total = df_total.drop('Attack', axis=1)

# Identificar las columnas numéricas y categóricas
num_cols = df_total.select_dtypes(include=['float64', 'int64']).columns
cat_cols = df_total.select_dtypes(include=['object']).columns

# Aplicar MinMaxScaler solo a las columnas numéricas
df_total[num_cols] = scaler.fit_transform(df_total[num_cols])

# Imprimir las características del dataframe transformado
print(f"Características: {df_total.columns.tolist()}")

# Lista para almacenar los valores F
f_values = []

# Realizar ANOVA para cada característica numérica
for caracteristica in num_cols:
    modelo = ols(f'{caracteristica} ~ C(dataset)', data=df_total).fit()
    resultado_anova = sm.stats.anova_lm(modelo, typ=2)
    f_value = resultado_anova.loc['C(dataset)', 'F']
    
    print(f"Valor F para {caracteristica}: {f_value}\n---\n")
    
    # Añadir el valor F a la lista
    f_values.append(f_value)

# Calcular la media de los valores F
f_mean = sum(f_values) / len(f_values)

print(f"La media de los valores F para este escenario es: {f_mean}")






