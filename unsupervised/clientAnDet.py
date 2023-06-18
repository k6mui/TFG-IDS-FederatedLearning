import flwr as fl
import tensorflow as tf
import pandas as pd
from pathlib import Path
import os
import sys
import numpy as np
from efc import EnergyBasedFlowClassifier
from utils import model, load_datasets
from server import serverAdress, eval_learningAnDet, anomalyDetection

# Make TensorFlow logs less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
# Parameter for std()
a = 1
with_efc = True


def calculate_MSE(x, x_hat):
    losses = np.mean((x - x_hat)**2, axis=1)  # MSE
    return losses

class FlwrClient(fl.client.NumPyClient):
    # Implement flower client extending class NumPyClient
    # This class serialize and de-serialize the weights into numpy ndarray

    def __init__(self, cid):
        self.cid = cid
        self.dataset = cid[cid.rfind("/") + 1:cid.find(".csv")]
        self.x_train, self.y_train, self.x_test, self.y_test = load_datasets.load_datasets(self.cid)

        if with_efc:
            self.model = model.create_NN(self.x_train.shape[1] + 1)  # consider one additional feature (EFC energy)
            efc = EnergyBasedFlowClassifier(cutoff_quantile=0.95)
            efc.fit(self.x_train, self.y_train)
            _, y_energies_train = efc.predict(self.x_train, return_energies=True)
            _, y_energies_test = efc.predict(self.x_test, return_energies=True)

            self.x_train = np.append(self.x_train, y_energies_train.reshape(y_energies_train.shape[0], 1), axis=1)
            self.x_test = np.append(self.x_test, y_energies_test.reshape(y_energies_test.shape[0], 1), axis=1)
        else:
            self.model = model.create_NN(self.x_train.shape[1])

        self.loss = 0
        self.threshold = 0
        self.trainSamples = self.x_train[self.y_train == 0] # only benign samples
        idx = int(self.trainSamples.shape[0] * 0.9)
        self.val_data = self.trainSamples[idx:]  # holdout validation set for threshold calculation
        self.train_data = self.trainSamples[:idx]  # reduced self.x_train (only benign wo val_data)

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
            self.train_data,
            self.train_data,
            batch_size=128,  # Configure the batch size
            shuffle=True,
            epochs=10, # Number of epochs
            validation_split = 0.1
        )

        self.loss = history.history["loss"][-1]

        return self.get_parameters(), len(self.x_train), {"loss": history.history["loss"][-1], }

    def evaluate(self, parameters, config):
        # evaluates the model on local data (locally)
        print(f"===========Client {self.cid} EVALUATE================")
        self.set_parameters(parameters)

        val_inference = self.model.predict(self.val_data)
        val_losses = calculate_MSE(self.val_data, val_inference)

        #Threshold Calculation#
        self.threshold = np.mean(val_losses)+a*np.std(val_losses)
        print("\n>> {} Mean Validation Loss: {} ".format(self.cid, self.threshold))

        # Test Set Evaluation
        inference = self.model.predict(self.x_test)
        losses = calculate_MSE(self.x_test, inference)

        test_eval = anomalyDetection(losses, self.threshold)

        acc, pre, rec, f1s, tn, fp, fn, tp = eval_learningAnDet(self.y_test, test_eval)


        output_dict = {"acc": acc, "rec": rec, "prec": pre, "f1": f1s, "tn": int(tn), "fp": int(fp),
                "fn": int(fn), "tp": int(tp)}

        print("\n>> Evaluate {} / Metrics {} / ".format(self.dataset, output_dict))
        
        return float(self.loss), len(self.x_test), output_dict


def main():
    # Start Flower client		
    if sys.argv[1] == "1":
        client = FlwrClient("./datasets/sampled/cicids_sampled.csv")
        fl.client.start_numpy_client(server_address=serverAdress, 
                                     client=client,
                                    #  root_certificates=Path(".cache/certificates/ca.crt").read_bytes()
                                     )
    elif sys.argv[1] == "2":
        client = FlwrClient("./datasets/sampled/nb15_sampled.csv")
        fl.client.start_numpy_client(server_address=serverAdress, 
                                     client=client,
                                    #  root_certificates=Path(".cache/certificates/ca.crt").read_bytes(),
                                     )
    elif sys.argv[1] == "3":
        client = FlwrClient("./datasets/sampled/toniot_sampled.csv")
        fl.client.start_numpy_client(server_address=serverAdress, 
                                     client=client,
                                    #  root_certificates=Path(".cache/certificates/ca.crt").read_bytes(),
                                     )
    else:
    # client does not exit
        print("Error: client does not exit")    

if __name__ == "__main__":
    main()
