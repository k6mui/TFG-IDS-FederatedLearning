import tensorflow as tf
from keras.models import Model, load_model
from keras.layers import Input, Dropout, Dense, Lambda
from tensorflow.keras.optimizers import SGD
from tensorflow.keras import optimizers

bcentr = "binary_crossentropy"
learning_rate = 0.001

def create_NN(num_features):
    """
        Creates a deep neural network to process a tabular data with 39 features 
    """
    model = tf.keras.models.Sequential([
        Input(shape=(num_features,)),
        Dense(256, activation='relu'),
        Dropout(0.3),
        Dense(256, activation='relu'),
        Dropout(0.3),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss=bcentr, metrics=['accuracy'])

    return model