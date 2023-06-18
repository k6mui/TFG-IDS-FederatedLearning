from xgutils import *

# Datasets can be downloaded from LIBSVM Data: https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/
binary_train = ["cod-rna.t", "cod-rna", "ijcnn1.t"]
binary_test = ["cod-rna.r", "cod-rna.t", "ijcnn1.tr"]
reg_train = ["eunite2001", "YearPredictionMSD"]
reg_test = ["eunite2001.t", "YearPredictionMSD.t"]

# Define the type of training task. Binary classification: BINARY; Regression: REG
task_types = ["BINARY", "REG"]
task_type = task_types[0]

# Select the downloaded training and test dataset
if task_type == "BINARY":
    dataset_path = "dataset/binary_classification/"
    train = binary_train[0]
    test = binary_test[0]
elif task_type == "REG":
    dataset_path = "dataset/regression/"
    train = reg_train[0]
    test = reg_test[0]

data_train = load_svmlight_file(dataset_path + train, zero_based=False)
data_test = load_svmlight_file(dataset_path + test, zero_based=False)

print("Task type selected is: " + task_type)
print("Training dataset is: " + train)
print("Test dataset is: " + test)

class TreeDataset(Dataset):
    def __init__(self, data: NDArray, labels: NDArray) -> None:
        self.labels = labels
        self.data = data

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Dict[int, NDArray]:
        label = self.labels[idx]
        data = self.data[idx, :]
        sample = {0: data, 1: label}
        return sample



