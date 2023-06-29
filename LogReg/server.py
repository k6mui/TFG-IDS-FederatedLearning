from typing import Optional, Dict, Tuple
import flwr as fl
import numpy as np
import sys
from sklearn.linear_model import LogisticRegression
import tensorflow as tf
from utils import centralizedTest, plot
from pathlib import Path
from sklearn.metrics import confusion_matrix, accuracy_score, log_loss, recall_score, precision_score, f1_score
import math
import matplotlib.pyplot as plt
import regUtils


# Ip of the central Server
serverAdress = "127.0.0.1:4687"
# Number of features 
NUM_FEATURES = 39

nrounds = 30

def eval_learning(model, X_test, Y_test, cid=""):
  y_pred = model.predict(X_test)
  loss = log_loss(Y_test, y_pred)
  acc = accuracy_score(y_pred, Y_test)
  pre = precision_score(y_pred, Y_test,zero_division = 0)
  rec = recall_score(y_pred, Y_test,zero_division = 0)
  f1s = f1_score(y_pred, Y_test,zero_division = 0)
  tn, fp, fn, tp = confusion_matrix(Y_test, y_pred).ravel()
    
  return loss, acc, pre, rec, f1s, tn, fp, fn, tp

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

# The `evaluate` function will be by Flower called after every round
def evaluate_REGLOG_CL(
    server_round: int,
    parameters: fl.common.NDArrays,
    config: Dict[str, fl.common.Scalar],
) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:
    # net = LogisticRegression(solver = 'lbfgs', max_iter=1, warm_start=True)
    # net = LogisticRegression(solver = 'saga', max_iter=1, warm_start=True, penalty='l1')
    model = LogisticRegression(random_state=10,
                               solver = 'sag',
                               max_iter=300,
                               warm_start=True
                               )
    X_test_CL, y_test_CL = centralizedTest.getTest()
    regUtils.set_initial_params(model) # First initialize parameters
    regUtils.set_model_params(model, parameters) # Update model with the latest parameters
    loss, accuracy, precision, recall, f1score  = eval_learning(model, X_test_CL, y_test_CL)
    print(f"@@@@@@ Server-side evaluation loss {loss} / accuracy {accuracy} / f1score {f1score} @@@@@@")
    return loss, {"accuracy": accuracy,"precision": precision,"recall": recall,"f1score": f1score}


def main():    

    # Create strategy
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1,
        fraction_evaluate=1,
        min_fit_clients=3,  
        min_evaluate_clients=3,  
        min_available_clients=3,
        evaluate_metrics_aggregation_fn=average_metrics,
        # on_fit_config_fn=fit_config,
        # initial_parameters=fl.common.ndarrays_to_parameters(params),

        # FedProx Parameters, regularitation term
        # proximal_mu  = 0.001
        
  
        # FedOpt Parameter, server learning rate FedYogi and FedAdam
        # eta = 0.05,
       
    )


    # Start Flower server
    history = fl.server.start_server(
            server_address=serverAdress,
            config=fl.server.ServerConfig(num_rounds=nrounds),
            strategy=strategy,
        certificates=(
            # Server will require a tuple of 3 certificates
            Path("../certificates/ca.crt").read_bytes(),
            Path("../certificates/server.pem").read_bytes(),
            Path("../certificates/server.key").read_bytes(),
    # )
    ))
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

    plt.grid(True)
    plt.show()