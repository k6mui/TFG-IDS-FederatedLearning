import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

seed=27

delete_features = ['IPV4_SRC_ADDR', 'L4_SRC_PORT', 'IPV4_DST_ADDR', 'L4_DST_PORT', 'Attack', 'Dataset', 'Label']

def remove_features(df, feats=delete_features): # -> X, y (dataframes)
    X = df.drop(columns=feats)
    y = df['Label']
    return X, y

def minMaxScaler(X, y, test_size): # -> train, test and index
    scaler = MinMaxScaler()
    indices = list(X.index)

    X_train, X_test, y_train, y_test, _, test_index = train_test_split(X, y, indices, test_size=test_size, 
    random_state=seed, stratify=y) # With the stratify option we get the same proportion of class distribution

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, y_train, X_test, y_test, test_index


def load_datasets(cid, test_size=0.2):
    df = pd.read_csv(cid, low_memory=True)
    df.dropna(inplace=True) # Make changes to original dataset
    print(">> \033[1m {} \033[0m - Total samples: {}, Benign: {}, Malicious: {}, Labels: {}"\
        .format(cid[cid.rfind("/")+1:cid.find(".csv")], df.shape[0], sum(df.Label == 0), \
        sum(df.Label == 1), sorted(list(df.Attack.unique().astype(str)))))
    X, y = remove_features(df)
    x_train, y_train, x_test, y_test, test_index = minMaxScaler(X, y, test_size)

    return x_train, y_train, x_test, y_test