import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

test_size=0.2
scaler = MinMaxScaler()
path1="../datasets/sampled/cicids_sampled.csv"
path2="../datasets/sampled/nb15_sampled.csv"
path3="../datasets/sampled/toniot_sampled.csv"

def rebalance_dataframe(df, label_col='Label', majority_class=0, minority_class=1, majority_target=700000, minority_target=300000):
    majority_df = df[df[label_col] == majority_class]
    minority_df = df[df[label_col] == minority_class]

    # Reducir o mantener la clase mayoritaria
    if len(majority_df) > majority_target:
        majority_df = majority_df.sample(n=majority_target, random_state=42)
    else:
        majority_df = majority_df.sample(n=majority_target, replace=True, random_state=42)

    # Reducir, mantener o duplicar la clase minoritaria
    if len(minority_df) > minority_target:
        minority_df = minority_df.sample(n=minority_target, random_state=42)
    else:
        minority_df = minority_df.sample(n=minority_target, replace=True, random_state=42)

    new_df = pd.concat([majority_df, minority_df])
    new_df = new_df.sample(frac=1, random_state=42)  # Mezcla los datos

    return new_df


def balance(path1=path1,path2=path2,path3=path3):

    df_cic = pd.read_csv(path1, low_memory=True)
    df_nb15 = pd.read_csv(path2, low_memory=True)
    df_toniot = pd.read_csv(path3, low_memory=True)

    print(">> \033[1m {} \033[0m - Total samples: {}, Benign: {}, Malicious: {}, Labels: {}"\
        .format(path1, df_cic.shape[0], sum(df_cic.Label == 0), \
        sum(df_cic.Label == 1), sorted(list(df_cic.Attack.unique().astype(str)))))
    
    print(">> \033[1m {} \033[0m - Total samples: {}, Benign: {}, Malicious: {}, Labels: {}"\
        .format(path2, df_nb15.shape[0], sum(df_nb15.Label == 0), \
        sum(df_nb15.Label == 1), sorted(list(df_nb15.Attack.unique().astype(str)))))
    
    print(">> \033[1m {} \033[0m - Total samples: {}, Benign: {}, Malicious: {}, Labels: {}"\
        .format(path3, df_toniot.shape[0], sum(df_toniot.Label == 0), \
        sum(df_toniot.Label == 1), sorted(list(df_toniot.Attack.unique().astype(str)))))

    print("-------------Rebalanceo----------------")
    # Rebalanceo de los dataframes
    df_cic = rebalance_dataframe(df_cic)
    df_nb15 = rebalance_dataframe(df_nb15)
    df_toniot = rebalance_dataframe(df_toniot)
   
    print(">> \033[1m {} \033[0m - Total samples: {}, Benign: {}, Malicious: {}, Labels: {}"\
        .format(path1, df_cic.shape[0], sum(df_cic.Label == 0), \
        sum(df_cic.Label == 1), sorted(list(df_cic.Attack.unique().astype(str)))))
    
    print(">> \033[1m {} \033[0m - Total samples: {}, Benign: {}, Malicious: {}, Labels: {}"\
        .format(path2, df_nb15.shape[0], sum(df_nb15.Label == 0), \
        sum(df_nb15.Label == 1), sorted(list(df_nb15.Attack.unique().astype(str)))))
    
    print(">> \033[1m {} \033[0m - Total samples: {}, Benign: {}, Malicious: {}, Labels: {}"\
        .format(path3, df_toniot.shape[0], sum(df_toniot.Label == 0), \
        sum(df_toniot.Label == 1), sorted(list(df_toniot.Attack.unique().astype(str)))))
    
    df_cic.to_csv('../datasets/balanced/cicids_balanced.csv', index=False)
    df_nb15.to_csv('../datasets/balanced/nb15_balanced.csv', index=False)
    df_toniot.to_csv('../datasets/balanced/toniot_balanced.csv', index=False)

if __name__ == "__main__":
    balance()
