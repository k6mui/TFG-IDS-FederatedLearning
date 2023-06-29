import pandas as pd
from sklearn.model_selection import train_test_split
import os

def load_and_combine_datasets(paths):
    try:
        dataframes = [pd.read_csv(path, low_memory=True) for path in paths]
        df_total = pd.concat(dataframes)
        return df_total
    except FileNotFoundError as e:
        print(f"Uno de los archivos no se encuentra en la ruta especificada: {e}")
        return None

def split_dataset(df, label_column='Attack'):
    X = df.drop(label_column, axis=1)
    y = df[label_column]

    # Divide el dataset en dos partes (66% para el conjunto temporal, 34% para el tercer conjunto)
    X_temp, X_third, y_temp, y_third = train_test_split(X, y, test_size=0.33, stratify=y, random_state=42)

    # Divide el conjunto temporal en dos para crear los primeros dos conjuntos
    X_first, X_second, y_first, y_second = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

    return pd.concat([X_first, y_first], axis=1), pd.concat([X_second, y_second], axis=1), pd.concat([X_third, y_third], axis=1)

def save_datasets(datasets, paths):
    for df, path in zip(datasets, paths):
        df.to_csv(path, index=False)

def print_label_distribution(datasets, names):
    for df, name in zip(datasets, names):
        print(f"\nDistribución de etiquetas en el DataFrame {name}:\n")
        print(df['Attack'].value_counts().to_string())

def balance(paths, save_paths, names):
    df_total = load_and_combine_datasets(paths)
    if df_total is None:
        return
    datasets = split_dataset(df_total)
    save_datasets(datasets, save_paths)
    print_label_distribution(datasets, names)

if __name__ == "__main__":
    paths = ["../datasets/sampled/cicids_sampled.csv", "../datasets/sampled/nb15_sampled.csv", "../datasets/sampled/toniot_sampled.csv"]
    save_paths = ['../datasets/balanced/first_balanced.csv', '../datasets/balanced/second_balanced.csv', '../datasets/balanced/third_balanced.csv']
    names = ["primero", "segundo", "tercero"]
    balance(paths, save_paths, names)
