import warnings
from pathlib import Path
import flwr as fl
import numpy as np
import os
import sys
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from server import eval_learning
import regUtils
from utils import load_datasets

# External IP of VM Instance
serverAdress = ""  # e.g. 34.125.18.50

class FlwrClient(fl.client.NumPyClient):
    # Implement flower client extending class NumPyClient
    # This class serialize and de-serialize the weights into numpy ndarray

    def __init__(self, cid):
        self.cid = cid
        self.dataset = cid[cid.rfind("/") + 1:cid.find(".csv")]
        self.x_train, self.y_train, self.x_test, self.y_test = load_datasets.load_datasets(self.cid)
        self.model = LogisticRegression(penalty='l2', solver = 'sag', max_iter=100, warm_start = True)
        regUtils.set_initial_params(self.model)

    def get_parameters(self, config):
        print(f"============Client {self.cid} GET_PARAMETERS===========")
        return regUtils.get_model_parameters(self.model)
     

    def fit(self, parameters, config):
        print(f"========Client {self.cid} FIT============")
        regUtils.set_model_params(self.model, parameters)
            # Ignore convergence failure due to low local epochs
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model.fit(self.x_train, self.y_train)

        return regUtils.get_model_parameters(self.model), len(self.x_train), {}

    def evaluate(self, parameters, config):
        # evaluates the model on local data (locally)
        print(f"===========Client {self.cid} EVALUATE================")
        regUtils.set_model_params(self.model, parameters)

        loss, accuracy, pre, rec, f1s, tn, fp, fn, tp = eval_learning(self.model, self.x_test, self.y_test, self.cid)
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
