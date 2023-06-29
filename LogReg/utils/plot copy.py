import pandas as pd
import matplotlib.pyplot as plt
import math
import numpy as np

loss1 = {
 1: 0.7420066666666667,
    2: 0.7539366666666667,
    3: 0.7405216666666666,
    4: 0.7449933333333334,
    5: 0.7565716666666668,
    6: 0.7634166666666667,
    7: 0.7635416666666667,
    8: 0.7617516666666667,
    9: 0.764725,
    10: 0.7650350000000001,
    11: 0.764755,
    12: 0.7670533333333333,
    13: 0.7744433333333333,
    14: 0.7762183333333333,
    15: 0.7737216666666665,
    16: 0.7759450000000001,
    17: 0.7764866666666667,
    18: 0.7775500000000001,
    19: 0.7835749999999999,
    20: 0.7832033333333334,
    21: 0.7809733333333333,
    22: 0.77827,
    23: 0.7754683333333334,
    24: 0.7736316666666667,
    25: 0.768275,
    26: 0.7665666666666667,
    27: 0.7671666666666667,
    28: 0.7668066666666666,
    29: 0.76668,
    30: 0.769285
    }
loss2 = {
  1: 0.7380933333333334,
    2: 0.7445033333333333,
    3: 0.746055,
    4: 0.7314216666666667,
    5: 0.7246,
    6: 0.7465383333333334,
    7: 0.7397266666666665,
    8: 0.7511766666666667,
    9: 0.7398033333333333,
    10: 0.7404716666666666,
    11: 0.7473933333333332,
    12: 0.7430783333333334,
    13: 0.745585,
    14: 0.747,
    15: 0.7465,
    16: 0.7474633333333333,
    17: 0.7509716666666666,
    18: 0.7777183333333334,
    19: 0.7847966666666667,
    20: 0.7866983333333333,
    21: 0.7930416666666668,
    22: 0.7943133333333333,
    23: 0.7942450000000001,
    24: 0.798865,
    25: 0.7979516666666666,
    26: 0.7994066666666667,
    27: 0.7924616666666666,
    28: 0.7909683333333334,
    29: 0.79224,
    30: 0.7919550000000001
}

    

loss3 = {  
  1: 0.621425,
    2: 0.6735150000000001,
    3: 0.7266699999999999,
    4: 0.77507,
    5: 0.77493,
    6: 0.7751583333333333,
    7: 0.726775,
    8: 0.7278533333333334,
    9: 0.702225,
    10: 0.7553916666666666,
    11: 0.7378516666666667,
    12: 0.7472150000000001,
    13: 0.7376900000000001,
    14: 0.7517616666666666,
    15: 0.7789566666666667,
    16: 0.7225116666666667,
    17: 0.7482033333333334,
    18: 0.7340399999999999,
    19: 0.7278033333333335,
    20: 0.7314483333333333,
    21: 0.7057333333333333,
    22: 0.744565,
    23: 0.7159016666666668,
    24: 0.7732333333333333,
    25: 0.7576583333333332,
    26: 0.767135,
    27: 0.7629983333333333,
    28: 0.7632183333333334,
    29: 0.7138766666666667,
    30: 0.725655
    }

loss4 = {
1: 0.6305166666666667,
    2: 0.7587183333333334,
    3: 0.7422166666666667,
    4: 0.6897716666666667,
    5: 0.66118,
    6: 0.7639316666666667,
    7: 0.7087983333333333,
    8: 0.6947683333333333,
    9: 0.7197883333333334,
    10: 0.71584,
    11: 0.7374616666666668,
    12: 0.7452666666666667,
    13: 0.696675,
    14: 0.702775,
    15: 0.7155283333333333,
    16: 0.7047116666666665,
    17: 0.7296699999999999,
    18: 0.7375783333333334,
    19: 0.7504233333333333,
    20: 0.7621349999999999,
    21: 0.7790533333333333,
    22: 0.756335,
    23: 0.7506400000000001,
    24: 0.7431733333333334,
    25: 0.7285666666666667,
    26: 0.7480633333333334,
    27: 0.7562533333333333,
    28: 0.7622666666666666,
    29: 0.7480866666666666,
    30: 0.7473483333333334
}

def plot_metric_from_history(
    hist1: dict,
    hist2: dict,
    hist3: dict,
    hist4: dict,
) -> None:
    """Function to plot from Flower server History.
    Parameters
    ----------
    hist : History
        Object containing evaluation for all rounds.
    save_plot_path : Path
        Folder to save the plot to.
    metric_type : Literal["centralized", "distributed"]
        Type of metric to plot.
    metric : Literal["accuracy","precision","recall","f1score"]
        Metric to plot.
    """
    
    rounds1 = list(hist1.keys())
    values1 = list(hist1.values())
    
    rounds2 = list(hist2.keys())
    values2 = list(hist2.values())
    rounds3 = list(hist3.keys())
    values3 = list(hist3.values())
    rounds4 = list(hist4.keys())
    values4 = list(hist4.values())

    # plt.plot(np.asarray(rounds), np.asarray(values), label="FedAvg")
    plt.plot(np.asarray(rounds1), np.asarray(values1), linewidth=1, label='FedAvg')
    plt.plot(np.asarray(rounds2), np.asarray(values2), linewidth=1, label='FedProx')
    plt.plot(np.asarray(rounds3), np.asarray(values3), linewidth=1, label='FedYogi')
    plt.plot(np.asarray(rounds4), np.asarray(values4), linewidth=1, label='FedAdam')
    plt.legend(fontsize=10)
    plt.xlabel('Communication round', fontsize=10)
    plt.ylabel("Accuracy", fontsize=10)
    plt.title("Accuracy", fontsize=10)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    # plt.ylim(min(min(min(commun_metrics))) - 0.05, max(max(max(commun_metrics))) + 0.05)
    plt.ylim(0.3, 0.9)
    plt.xlim(1, 30)

    # plt.savefig(Path(save_plot_path) / Path(f"{metric_type}_metrics{suffix}.png"))
    # plt.close()

def plot_losses_from_history(
    loss1: dict,
    loss2: dict,
    loss3: dict,
    loss4: dict
) -> None:
    """Function to plot from Flower server History.
    Parameters
    ----------
    hist : History
        Object containing evaluation for all rounds.
    save_plot_path : Path
        Folder to save the plot to.
    metric_type : Literal["centralized", "distributed"]
        Type of metric to plot.
    metric : Literal["accuracy","precision","recall","f1score"]
        Metric to plot.
    """
    
    rounds1 = list(loss1.keys())
    values1 = list(loss1.values())

    rounds2 = list(loss2.keys())
    values2 = list(loss2.values())

    rounds3 = list(loss3.keys())
    values3 = list(loss3.values())

    rounds4 = list(loss4.keys())
    values4 = list(loss4.values())

    # plt.plot(np.asarray(rounds), np.asarray(values), label="FedAvg")
    plt.plot(np.asarray(rounds1), np.asarray(values1), linewidth=1, label='FedAvg')
    plt.plot(np.asarray(rounds2), np.asarray(values2), linewidth=1, label='FedProx')
    plt.plot(np.asarray(rounds3), np.asarray(values3), linewidth=1, label='FedYogi')
    plt.plot(np.asarray(rounds4), np.asarray(values4), linewidth=1, label='FedAdam')

    plt.legend(fontsize=10)
    plt.xlabel('Communication round', fontsize=10)
    plt.ylabel("loss", fontsize=10)
    plt.title("Pérdidas", fontsize=10)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    # plt.ylim(min(min(min(commun_metrics))) - 0.05, max(max(max(commun_metrics))) + 0.05)
    plt.ylim(0, 2)
    plt.xlim(1, 30)

    # plt.savefig(Path(save_plot_path) / Path(f"{metric_type}_metrics{suffix}.png"))
    # plt.close()

def main():
    # plot_losses_from_history(loss1, loss2, loss3, loss4)
    metrics_show = ["accuracy","precision","recall","f1score"]

    plot_metric_from_history(loss1, loss2, loss3, loss4)
    plt.grid(True)
    plt.show()
if __name__ == "__main__":
    main()