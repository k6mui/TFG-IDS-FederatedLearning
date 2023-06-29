import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from utils import load_datasets

test_size=0.2
scaler = MinMaxScaler()
path1="./datasets/sampled/cicids_sampled.csv"
path2="./datasets/sampled/nb15_sampled.csv"
path3="./datasets/sampled/toniot_sampled.csv"

def getTest(path1=path1,path2=path2,path3=path3):

    df_cic = pd.read_csv(path1, low_memory=True)
    df_nb15 = pd.read_csv(path2, low_memory=True)
    df_toniot = pd.read_csv(path3, low_memory=True)

    df_CL = pd.concat([df_cic, df_nb15, df_toniot]).reset_index(drop=True)
    X_CL, Y_CL = load_datasets.remove_features(df_CL)

    # Split data into train and test
    _, X_test_CL, _, y_test_CL = train_test_split(X_CL, Y_CL,shuffle=True, test_size=0.2, random_state=42, stratify=Y_CL)
    _, _, X_test_CL, y_test_CL, test_index = load_datasets.minMaxScaler(X_CL, Y_CL, test_size)

    scaler.fit(X_test_CL)
    # X_train_CL = scaler.transform(X_train_CL)
    X_test_CL = scaler.transform(X_test_CL)
    
    return X_test_CL, y_test_CL