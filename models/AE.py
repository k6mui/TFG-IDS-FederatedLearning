import tensorflow as tf
from keras.models import Model, load_model
from keras.layers import Input, Dropout, Dense, Lambda
from tensorflow.keras.optimizers import SGD
from tensorflow.keras import optimizers

MAE = "mae"
learning_rate = 0.001

def build_encoder(input_layer):
    """
    Builds the encoder part of the autoencoder.
    """
    encoded = Dense(64, activation='relu')(input_layer)
    encoded = Dense(32, activation='relu')(encoded)
    encoded = Dense(16, activation='relu')(encoded)
    
    return encoded

def build_bottleneck(encoded_layer):
    """
    Builds the bottleneck part of the autoencoder.
    """
    bottleneck = Dense(8, activation='relu')(encoded_layer)
    return bottleneck

def build_decoder(bottleneck_layer, num_features):
    """
    Builds the decoder part of the autoencoder.
    """
    decoded = Dense(16, activation='relu')(bottleneck_layer)
    decoded = Dense(32, activation='relu')(decoded)
    decoded = Dense(64, activation='relu')(decoded)
    decoded = Dense(num_features, activation='sigmoid')(decoded)
    return decoded

def compile_autoencoder(input_layer, output_layer):
    """
    Compiles the autoencoder model.
    """
    autoencoder = Model(inputs=input_layer, outputs=output_layer)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    autoencoder.compile(optimizer=optimizer, loss=MAE)
    return autoencoder

def create_AE(num_features):
    """
    Creates a deep autoencoder neural network to process tabular data with 'num_features' features.
    """
    # Input layer
    input_layer = Input(shape=(num_features,))
    
    # Encoder
    encoded_layer = build_encoder(input_layer)

    # Bottleneck
    bottleneck_layer = build_bottleneck(encoded_layer)

    # Decoder
    decoded_layer = build_decoder(bottleneck_layer, num_features)

    # Compile the Model
    autoencoder = compile_autoencoder(input_layer, decoded_layer)
    
    return autoencoder