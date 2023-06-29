from typing import Optional, Dict, Tuple
import flwr as fl
import numpy as np
import tensorflow as tf
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../")))
from utils import centralizedTest,  load_datasets, plot
from models import AE
from pathlib import Path
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score
import math
import matplotlib.pyplot as plt

# Ip of the central Server
serverAdress = "127.0.0.1:4687"

# Number of features 
NUM_FEATURES = 39

# Number of rounds
nrounds = 30


# # For Autoencoders
def anomalyDetection(losses, threshold):
    predictions = np.zeros(len(losses))
    for i, MAE in enumerate(losses):
        if MAE > threshold:
            predictions[i] = 1
        else:
            predictions[i] = 0

    return predictions

def eval_learningAnDet(Y_test, y_pred):
  acc = accuracy_score(Y_test, y_pred)
  tn, fp, fn, tp = confusion_matrix(Y_test, y_pred).ravel()
  acc = accuracy_score(y_pred, Y_test)
  pre = precision_score(y_pred, Y_test ,zero_division = 0)
  rec = recall_score(y_pred, Y_test, zero_division = 0)
  f1s = f1_score(y_pred, Y_test ,zero_division = 0)
    
  return acc, pre, rec, f1s, tn, fp, fn, tp

def average_metrics(metrics):
    accuracies = [metric["accuracy"] for _, metric in metrics]
    recalls = [metric["recall"] for _, metric in metrics]
    precisions = [metric["precision"] for _, metric in metrics]
    f1s = [metric["f1score"] for _, metric in metrics]
    fpr = [metric["fpr"] for _, metric in metrics]
    

    accuracies = sum(accuracies) / len(accuracies)
    recalls = sum(recalls) / len(recalls)
    precisions = sum(precisions) / len(precisions)
    f1s = sum(f1s) / len(f1s)
    fpr = sum(fpr) / len(fpr)
    

    return {"accuracy": accuracies, "recall": recalls, "precision": precisions, "f1score": f1s, "fpr":fpr}

# def evaluate_DNN_CL(
#     server_round: int,
#     parameters: fl. common.NDArrays,
#     config: Dict[str, fl.common.Scalar],
# ) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:
#     model = LSTM.create_NN(39)
#     X_test_CL, y_test_CL = centralizedTest.getTest()
#     model.set_weights(parameters) # Update model with the latest parameters
#     loss, accuracy, precision, recall, f1score,tn, fp, fn, tp  = eval_learning(model, X_test_CL, y_test_CL)
#     print(f"@@@@@@ Server-side evaluation loss {loss} / accuracy {accuracy} / f1score {f1score} @@@@@@")
#     return loss, {"accuracy": accuracy,"precision": precision,"recall": recall,"f1score": f1score, "tn": tn, "fp": fp, "fn": fn, "tp": tp}



def main():    
    
    params = AE.create_AE(NUM_FEATURES).get_weights()

    # Create strategy
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1,
        fraction_evaluate=1,
        min_fit_clients=3,  
        min_evaluate_clients=3,  
        min_available_clients=3,
        evaluate_metrics_aggregation_fn=average_metrics,
        # evaluate_fn=evaluate_DNN_CL,
        # on_fit_config_fn=fit_config,
        initial_parameters=fl.common.ndarrays_to_parameters(params),

        # FedProx Parameters, regularitation term
        # proximal_mu  = 1,
        
  
        # FedOpt Parameter, server learning rate
        # eta = 0.05,
       
    )


    # Start Flower server
    history = fl.server.start_server(
            server_address=serverAdress,
            config=fl.server.ServerConfig(num_rounds=nrounds),
            strategy=strategy,
            certificates=(  # Server will require a tuple of 3 certificates
                Path("../certificates/ca.crt").read_bytes(),
                Path("../certificates/server.pem").read_bytes(),
                Path("../certificates/server.key").read_bytes()
                )
            )
    
    
    return history



if __name__ == "__main__":

    history = main()
    global_acc_test = plot.retrieve_global_metrics(history,"distributed","accuracy",False)
    global_pre_test = plot.retrieve_global_metrics(history,"distributed","precision",False)
    global_rec_test = plot.retrieve_global_metrics(history,"distributed","recall",False)
    global_f1s_test = plot.retrieve_global_metrics(history,"distributed","f1score",False)
    global_fpr_test = plot.retrieve_global_metrics(history,"distributed","fpr",False)
    global_fnr_test = plot.retrieve_global_metrics(history,"distributed","fnr",False)



    print("\n\nFINAL RESULTS: ===========================================================================================================================================================================================")
    print('Test:  global_acc: {:} | global_pre: {} | global_rec: {} | global_f1s: {} | global_fpr: {} | global_fnr: {}'.format(global_acc_test, global_pre_test, global_rec_test, global_f1s_test, global_fpr_test, global_fnr_test))

    plot.plot_losses_from_history(history, "any", "distributed", )
    # Define metrics to plot
    
    plt.grid(True)
    plt.show()