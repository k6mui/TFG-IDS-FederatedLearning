import flwr as fl
import tensorflow as tf
import pandas as pd
from pathlib import Path
import os
import sys
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score, matthews_corrcoef, roc_auc_score
from utils import model, load_datasets
from server import serverAdress

# Make TensorFlow logs less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def eval_learning(model, X_test, Y_test):
  bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)
  logits = model.predict(X_test, batch_size=32, verbose=1)
  y_pred = logits
  y_pred[y_pred <= 0.5] = 0.
  y_pred[y_pred > 0.5] = 1.
  tn, fp, fn, tp = confusion_matrix(Y_test, y_pred).ravel()
  loss = bce(Y_test, logits.reshape((len(logits),))).numpy()
  acc = accuracy_score(y_pred, Y_test)
  pre = precision_score(y_pred, Y_test,zero_division = 0)
  rec = recall_score(y_pred, Y_test, zero_division = 0)
  f1s = f1_score(y_pred, Y_test, zero_division = 0)
    
  return loss, acc, pre, rec, f1s, tn, fp, fn, tp

class FlwrClient(fl.client.NumPyClient):
    # Implement flower client extending class NumPyClient
    # This class serialize and de-serialize the weights into numpy ndarray

    def __init__(self, cid):
        self.cid = cid
        self.dataset = cid[cid.rfind("/") + 1:cid.find(".csv")]
        self.x_train, self.y_train, self.x_test, self.y_test = load_datasets.load_datasets(self.cid)
        self.model = model.create_NN(self.x_train.shape[1])

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
            epochs=10, # Number of epochs
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
        _, acc, pre, rec, f1s, tn, fp, fn, tp = eval_learning(self.model, self.x_test, self.y_test)


        output_dict = {"acc": accuracy, "rec": rec, "prec": pre, "f1": f1s, "tn": int(tn), "fp": int(fp),
                "fn": int(fn), "tp": int(tp)}

        print("\n>> Evaluate {} / Metrics {} / ".format(self.dataset, output_dict))
        
        return loss, len(self.x_test), output_dict


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
