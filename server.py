from typing import Optional, Dict, Tuple
import flwr as fl
import numpy as np
import tensorflow as tf
from utils import test,  load_datasets
from models import dnn
from pathlib import Path
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score


# Ip of the central Server
serverAdress = "127.0.0.1:4687"
# Number of features 
NUM_FEATURES = 39
# Use EFC
with_efc = False

nrounds = 50


# # # For Autoencoders
# def anomalyDetection(losses, threshold):
#     predictions = np.zeros(len(losses))
#     for i, mse in enumerate(losses):
#         if mse > threshold:
#             predictions[i] = 1
#         else:
#             predictions[i] = 0

#     return predictions

# def eval_learningAnDet(Y_test, y_pred):
#   acc = accuracy_score(Y_test, y_pred)
#   tn, fp, fn, tp = confusion_matrix(Y_test, y_pred).ravel()
#   acc = accuracy_score(y_pred, Y_test)
#   pre = precision_score(y_pred, Y_test,zero_division = 0)
#   rec = recall_score(y_pred, Y_test, zero_division = 0)
#   f1s = f1_score(y_pred, Y_test, zero_division = 0)
    
#   return acc, pre, rec, f1s, tn, fp, fn, tp

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

def average_metrics(metrics):
    accuracies = [metric["acc"] for _, metric in metrics]
    recalls = [metric["rec"] for _, metric in metrics]
    precisions = [metric["prec"] for _, metric in metrics]
    f1s = [metric["f1"] for _, metric in metrics]
    

    accuracies = sum(accuracies) / len(accuracies)
    recalls = sum(recalls) / len(recalls)
    precisions = sum(precisions) / len(precisions)
    f1s = sum(f1s) / len(f1s)
    

    return {"acc": accuracies, "rec": recalls, "prec": precisions, "f1": f1s}

def evaluate_DNN_CL(
    server_round: int,
    parameters: fl. common.NDArrays,
    config: Dict[str, fl.common.Scalar],
) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:
    net = dnn.create_NN(39)
    X_test_CL, y_test_CL = test.getTest()
    net.set_weights(parameters) # Update model with the latest parameters
    loss, accuracy, precision, recall, f1score,tn, fp, fn, tp  = eval_learning(net, X_test_CL, y_test_CL)
    print(f"@@@@@@ Server-side evaluation loss {loss} / accuracy {accuracy} / f1score {f1score} @@@@@@")
    return loss, {"accuracy": accuracy,"precision": precision,"recall": recall,"f1score": f1score, "tn": tn, "fp": fp, "fn": fn, "tp": tp}

def main():    
    if with_efc:
        print("uso EFC")
        params = dnn.create_NN(NUM_FEATURES + 1).get_weights()  # Additional Feature if using EFC
    else:
        params = dnn.create_NN(NUM_FEATURES).get_weights()

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
        initial_parameters=fl.common.ndarrays_to_parameters(params)

        # FedProx Parameters
        # proximal_mu  = 0.1
        
        # FedAvgM Parameters
        # server_learning_rate=1.0,
        # server_momentum=0.95,

        # FedOpt Parameters
        # eta = 1e-1,
        # eta_l = 1e-1,
        # beta_1 = 0.0,
        # beta_2 = 0.0,
        # tau = 1e-9,
    )

    # Start Flower server
    fl.server.start_server(
            server_address=serverAdress,
            config=fl.server.ServerConfig(num_rounds=nrounds),
            strategy=strategy,
        certificates=(
            # Server will require a tuple of 3 certificates
            Path("./certificates/cache/certificates/ca.crt").read_bytes(),
            Path("./certificates/cache/certificates/server.pem").read_bytes(),
            Path("./certificates/cache/certificates/server.key").read_bytes(),
    )
    )


if __name__ == "__main__":
    main()