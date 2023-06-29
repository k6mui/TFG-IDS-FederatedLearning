import flwr as fl
import tensorflow as tf
import pandas as pd
from pathlib import Path
import os
import sys
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../")))
from utils import load_datasets
from models import dnn
from server import eval_learning

# External IP of VM Instance
serverAdress = ""  # e.g. 34.125.18.50

# Make TensorFlow logs less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

class FlwrClient(fl.client.NumPyClient):
    # Implement flower client extending class NumPyClient
    # This class serialize and de-serialize the weights into numpy ndarray

    def __init__(self, cid):
        self.cid = cid
        self.dataset = cid[cid.rfind("/") + 1:cid.find(".csv")]
        self.x_train, self.y_train, self.x_test, self.y_test = load_datasets.load_datasets(self.cid)
        self.model = dnn.create_NN(self.x_train.shape[1])
        
    def get_parameters(self):
        print(f"============Client {self.cid} GET_PARAMETERS===========")
        return self.model.get_weights()  # get_weights from keras returns the weights as ndarray

    def set_parameters(self, parameters):
        # Server sets model parameters from a list of NumPy ndarrays (Optional)
        self.model.set_weights(parameters)  # set_weights on local model (similar to get_weights)

    def fit(self, parameters, config):
        print(f"========Client {self.cid} FIT============")
        self.set_parameters(parameters)

        history = self.model.fit(
            self.x_train,
            self.y_train,
            batch_size=128,  # Configure the batch size
            shuffle=True,
            epochs=5, # Number of epochs
            validation_split = 0.1
        )

        self.loss = history.history["loss"][-1]

        return self.get_parameters(), len(self.x_train), {"loss": history.history["loss"][-1], }

    def evaluate(self, parameters, config):
        # evaluates the model on local data (locally)
        print(f"===========Client {self.cid} EVALUATE================")
        self.set_parameters(parameters)

        # Test Set Evaluation
        loss, accuracy = self.model.evaluate(self.x_test, self.y_test, verbose=2)
        _, _, pre, rec, f1s, tn, fp, fn, tp = eval_learning(self.model, self.x_test, self.y_test, self.cid)
        fpr = fp/(tn+fp)

        output_dict = {"accuracy": accuracy, "recall": rec, "precision": pre, "f1score": f1s, "fpr": fpr}

        print("\n>> Evaluate {} / Metrics {} / ".format(self.dataset, output_dict))
        
        return loss, len(self.x_test), output_dict


def main():
    # Start Flower client		
    if sys.argv[1] == "1":
        client = FlwrClient("..\datasets\sampled\cicids_sampled.csv")
        fl.client.start_numpy_client(server_address=serverAdress, 
                                     client=client,
                                     root_certificates=Path("../certificates/ca.crt").read_bytes()
                                     )
    elif sys.argv[1] == "2":
        client = FlwrClient("..\datasets\sampled/nb15_sampled.csv")
        fl.client.start_numpy_client(server_address=serverAdress, 
                                     client=client,
                                     root_certificates=Path("../certificates/ca.crt").read_bytes(),
                                     )
    elif sys.argv[1] == "3":
        client = FlwrClient("..\datasets\sampled/toniot_sampled.csv")
        fl.client.start_numpy_client(server_address=serverAdress, 
                                     client=client,
                                     root_certificates=Path("../certificates/ca.crt").read_bytes(),
                                     )
    else:
    # client does not exit
        print("Error: client does not exit")    

if __name__ == "__main__":
    main()
