import flwr as fl
from utils import model, load_datasets
from pathlib import Path

# Ip of the central Server
serverAdress = "34.175.241.238:4687"

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


def main():    
    sample_silo = "./datasets/sampled/nb15_sampled.csv"
    x_train, _, _, _ = load_datasets.load_datasets(sample_silo)
    
    params = model.create_NN(x_train.shape[1]).get_weights()

    del x_train # release memory

    # Create strategy
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1,
        fraction_evaluate=1,
        min_available_clients=3,
        evaluate_metrics_aggregation_fn=average_metrics,
        # eval_fn=centralized_eval,
        # on_fit_config_fn=fit_config,
        initial_parameters=fl.common.ndarrays_to_parameters(params)

        # FedProx Parameters
        # proximal_mu  = 1
        
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
            config=fl.server.ServerConfig(num_rounds=10),
            strategy=strategy,
    #     ,certificates=(
    #         # Server will require a tuple of 3 certificates
    #         Path(".cache/certificates/ca.crt").read_bytes(),
    #         Path(".cache/certificates/server.pem").read_bytes(),
    #         Path(".cache/certificates/server.key").read_bytes(),
    # )
    )


if __name__ == "__main__":
    main()