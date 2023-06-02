import tensorflow as tf
from keras.models import Model, load_model
from keras.layers import Input, Dropout, Dense, Lambda
from tensorflow.keras import optimizers
from tensorflow.keras import backend as K

def create_NN(num_features):
    """
        Creates a deep neural network to process a tabular data with 39 features 
    """
    model = tf.keras.models.Sequential([
        Input(shape=(num_features,)),
        Dense(12, activation='relu'),
        Dense(6, activation='relu'),
        Dense(3, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=['accuracy'])
    
    return model