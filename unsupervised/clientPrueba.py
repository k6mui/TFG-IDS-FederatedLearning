import flwr as fl
import tensorflow as tf
import pandas as pd
from pathlib import Path
import os
import sys
import numpy as np
from efc import EnergyBasedFlowClassifier
from utils import model, load_datasets
from server import serverAdress, eval_learningAnDet, with_efc

# Make TensorFlow logs less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"



def distance_calc(losses, benign, attack):
    # for each sample loss, calculate the minimun distance and set a label for test purpose
    result = np.zeros(len(losses))
    for i, loss in enumerate(losses):
        if abs(loss - benign) > abs(loss - attack):
            result[i] = 1
        else:
            result[i] = 0

    return result


def calculate_reconstruction_loss(x, x_hat):
    losses = np.mean(abs(x - x_hat), axis=1)  # MAE
    return losses


class FlwrClient(fl.client.NumPyClient):
    # Implement flower client extending class NumPyClient
    # This class serialize and de-serialize the weights into numpy ndarray

    def __init__(self, cid):
        self.cid = cid
        self.pre_cid = cid[cid.rfind("/") + 1:cid.find(".csv")]
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
        self.threshold_benign = 0  # this threshold is calculated o % benign samples from train data (during evaluate)
        self.threshold_attack = 0  # this threshold is calculated on attack samples from train data (during evaluate)

        train_data = self.x_train[self.y_train == 0]  # only benign samples
        idx = int(train_data.shape[0] * 0.9)
        self.val_data = train_data[idx:]  # holdout validation set for threshold calculation
        self.train_data = train_data[:idx]  # reduced self.x_train (only benign wo val_data)
        self.attack_data = self.x_train[
            self.y_train == 1]  # the attack data from train set (used only for threshold estimation)

    def get_parameters(self):
        # return local model parameters
        return self.model.get_weights()  # get_weights from keras returns the weights as ndarray

    def set_parameters(self, parameters):
        # Server sets model parameters from a list of NumPy ndarrays (Optional)
        self.model.set_weights(parameters)  # set_weights on local model (similar to get_weights)

    def fit(self, parameters, config):
        # receive parameters from the server and use them to train on local data (locally)
        self.set_parameters(parameters)

        # https://keras.io/api/models/model_training_apis/#fit-method
        history = self.model.fit(
            self.train_data,
            self.train_data,
            batch_size=128,  # config["batch_size"],
            shuffle=True,
            epochs=10  # config["num_epochs"] # single epoch on local data
        )

        # return the refined model parameters with get_weights, the length of local data,
        #        and custom metrics can be provided through dict
        # len(x_train) is a useful information for FL, analogous to weights of contribution of each client
        self.loss = history.history["loss"][-1]

        return self.get_parameters(), len(self.train_data), {"loss": history.history["loss"][-1], }

    def evaluate(self, parameters, config):
        # evaluates the model on local data (locally)
        self.set_parameters(parameters)

        # eval new model on holdout validation set
        val_inference = self.model.predict(self.val_data)
        attack_inference = self.model.predict(self.attack_data)
        val_losses = calculate_reconstruction_loss(self.val_data, val_inference)
        attack_losses = calculate_reconstruction_loss(self.attack_data, attack_inference)

        #########################
        # Threshold Calculation #
        #########################
        self.threshold_benign = np.mean(val_losses)
        # self.threshold_benign = np.quantile(val_losses, 0.95)
        self.threshold_attack = np.mean(attack_losses)

        print("\n>> {} Mean Validation Loss (Benign): {} | (Attack): {}".format(self.pre_cid, self.threshold_benign,
                                                                                self.threshold_attack))

        # Test Set Evaluation
        inference = self.model.predict(self.x_test)
        losses = calculate_reconstruction_loss(self.x_test, inference)

        ######################
        # Threshold Criteria #
        ######################
        test_eval = distance_calc(losses, self.threshold_benign, self.threshold_attack) # this considers both benign and attack thresholds
        # test_eval = losses > self.threshold_benign  # this considers only the benign threshold (comment out this line to use this criteria)

        acc, pre, rec, f1s, tn, fp, fn, tp = eval_learningAnDet(self.y_test, test_eval)


        output_dict = {"acc": acc, "rec": rec, "prec": pre, "f1": f1s, "tn": int(tn), "fp": int(fp),
                "fn": int(fn), "tp": int(tp)}

        print("\n>> Evaluate {} / Metrics {} / ".format(self.cid, output_dict))
        
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
