import tensorflow as tf
from keras.models import Model, load_model
from keras.layers import Input, Dropout, Dense, Lambda
from tensorflow.keras import optimizers
from tensorflow.keras import backend as K

MSE = "mean_squared_error"
bcentr = "binary_crossentropy"

def create_NN(num_features):
    """
        Creates a deep neural network to process a tabular data with 39 features 
    """
    model = tf.keras.models.Sequential([
        Input(shape=(num_features,)),
        Dense(256, activation='relu'),
        Dropout(0.2),
        Dense(256, activation='relu'),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer="adam", loss=bcentr, metrics=['accuracy'])

    # model = tf.keras.models.Sequential([
    #     Input(shape=(num_features,)),
    #     Dense(32, activation='relu'),
    #     Dense(16, activation='relu'),
    #     Dense(8, activation='relu'),
    #     Dense(4, activation='relu'),
    #     Dense(8, activation='relu'),
    #     Dense(16, activation='relu'),
    #     Dense(32, activation='relu'),
    #     Dense(num_features, activation='sigmoid')
    # ])
    
    # model.compile(optimizer="adam", loss=MSE)
    
    return model