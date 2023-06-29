import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler

# Create an instance of LabelEncoder and MinMaxScaler
label_encoder = LabelEncoder()
scaler = MinMaxScaler()

# Features to be deleted
delete_features = ['IPV4_SRC_ADDR', 'L4_SRC_PORT', 'IPV4_DST_ADDR', 'L4_DST_PORT', 'Dataset']

# Paths to the CSV files for the heterogeneous scenario
# paths = [
#     "../datasets/sampled/cicids_sampled.csv",
#     "../datasets/sampled/nb15_sampled.csv",
#     "../datasets/sampled/toniot_sampled.csv"
# ]

# Paths to the CSV files for the non-heterogeneous scenario
paths = [
    "../datasets/balanced/first_balanced.csv",
    "../datasets/balanced/second_balanced.csv",
    "../datasets/balanced/third_balanced.csv"
]

# Dataset names for the new 'dataset' column
dataset_names = ['CIC', 'NB15', 'TONIOT']

# Number of rows to sample from each DataFrame
numero_filas_muestreo = 500000

# List to store the sampled DataFrames
sampled_dfs = []

# Load the CSV files, sample rows, delete unnecessary features, and add the 'dataset' column
for path, dataset_name in zip(paths, dataset_names):
    df = pd.read_csv(path, low_memory=False)
    df_sampled = df.sample(n=numero_filas_muestreo, random_state=82)
    df_sampled = df_sampled.drop(columns=delete_features)
    df_sampled['dataset'] = dataset_name
    sampled_dfs.append(df_sampled)

# Combine the three sampled DataFrames into one
df_total = pd.concat(sampled_dfs)

# Encode the 'Attack' column and then drop it
df_total['Attack_encoded'] = label_encoder.fit_transform(df_total['Attack'])
df_total = df_total.drop('Attack', axis=1)

# Identify the numeric and categorical columns
num_cols = df_total.select_dtypes(include=['float64', 'int64']).columns
cat_cols = df_total.select_dtypes(include=['object']).columns

# Apply MinMaxScaler only to the numeric columns
df_total[num_cols] = scaler.fit_transform(df_total[num_cols])

# Print the features of the transformed dataframe
print(f"Features: {df_total.columns.tolist()}")

# List to store the F-values
f_values = []

# Perform ANOVA for each numeric feature
for feature in num_cols:
    model = ols(f'{feature} ~ C(dataset)', data=df_total).fit()
    anova_result = sm.stats.anova_lm(model, typ=2)
    f_value = anova_result.loc['C(dataset)', 'F']
    
    print(f"F-value for {feature}: {f_value}\n---\n")
    
    # Add the F-value to the list
    f_values.append(f_value)

# Calculate the mean of the F-values
f_mean = sum(f_values) / len(f_values)

print(f"The mean of the F-values for this scenario is: {f_mean}")







