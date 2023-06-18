from loaddata import *
from xgutils import *
from utils import load_datasets

# The number of clients participated in the federated learning
client_num = 3

# The number of XGBoost trees in the tree ensemble that will be built for each client
client_tree_num = 500 // client_num

global_tree = construct_tree(X_train, y_train, client_tree_num, task_type)
preds_train = global_tree.predict(X_train)
preds_test = global_tree.predict(X_test)


result_train = accuracy_score(y_train, preds_train)
result_test = accuracy_score(y_test, preds_test)
print("Global XGBoost Training Accuracy: %f" % (result_train))
print("Global XGBoost Testing Accuracy: %f" % (result_test))
