import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from pathlib import Path
import os
import sys
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../")))
from utils import load_datasets
from models import dnn, AE
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

test_size = 0.2
scaler = MinMaxScaler()

path1 = "../datasets/sampled/cicids_sampled.csv"
path2 = "../datasets/sampled/nb15_sampled.csv"
path3 = "../datasets/sampled/toniot_sampled.csv"

# Cargar y preprocesar el primer dataset (CICIDS)
x_trainCic, y_trainCic, x_testCic, y_testCic = load_datasets.load_datasets(path1)
x_trainCic = scaler.fit_transform(x_trainCic)
x_testCic = scaler.transform(x_testCic)

# Cargar y preprocesar el segundo dataset (NB15)
x_trainNb15, y_trainNb15, x_testNb15, y_testNb15 = load_datasets.load_datasets(path2)
x_trainNb15 = scaler.fit_transform(x_trainNb15)
x_testNb15 = scaler.transform(x_testNb15)

# Cargar y preprocesar el tercer dataset (Toniot)
x_trainTon, y_trainTon, x_testTon, y_testTon = load_datasets.load_datasets(path3)
x_trainTon = scaler.fit_transform(x_trainTon)
x_testTon = scaler.transform(x_testTon)

# Crear y entrenar el modelo
model = dnn.create_NN(num_features=x_trainNb15.shape[1])
history = model.fit(x_trainCic, y_trainCic, epochs=100, batch_size=128, verbose=1, validation_split = 0.1)

# Training plot
plt.figure(figsize=(8,6))
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(fontsize=10)
plt.xlabel('Epochs', fontsize=10)
plt.ylabel("Loss", fontsize=10)
plt.title("Training Loss", fontsize=10)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.xlim(0, 9)
plt.legend(['Train', 'Validation'], loc='upper right')
plt.grid(True)
plt.show()

# Evaluar en el conjunto de test de NB15
y_predNb15 = model.predict(x_testNb15)
y_predNb15 = np.round(y_predNb15)
accuracyNb15 = accuracy_score(y_testNb15, y_predNb15)
precisionNb15 = precision_score(y_testNb15, y_predNb15)
recallNb15 = recall_score(y_testNb15, y_predNb15)
f1scoreNb15 = f1_score(y_testNb15, y_predNb15)
tn, fp, fn, tp = confusion_matrix(y_testNb15, y_predNb15).ravel()
fpr_nb15 = fp / (fp + tn)

print("Metrics for NB15 dataset:")
print("Accuracy:", accuracyNb15)
print("Precision:", precisionNb15)
print("Recall:", recallNb15)
print("F1-Score:", f1scoreNb15)
print("FPR for NB15 dataset:", fpr_nb15)

# Evaluar en el conjunto de test de CICIDS
y_predCic = model.predict(x_testCic)
y_predCic = np.round(y_predCic)
accuracyCic = accuracy_score(y_testCic, y_predCic)
precisionCic = precision_score(y_testCic, y_predCic)
recallCic = recall_score(y_testCic, y_predCic)
f1scoreCic = f1_score(y_testCic, y_predCic)
tn, fp, fn, tp = confusion_matrix(y_testCic, y_predCic).ravel()
fpr_cic = fp / (fp + tn)

print("Metrics for CICIDS dataset:")
print("Accuracy:", accuracyCic)
print("Precision:", precisionCic)
print("Recall:", recallCic)
print("F1-Score:", f1scoreCic)
print("FPR for CICIDS dataset:", fpr_cic)

# Evaluar en el conjunto de test de Toniot
y_predTon = model.predict(x_testTon)
y_predTon = np.round(y_predTon)
accuracyTon = accuracy_score(y_testTon, y_predTon)
precisionTon = precision_score(y_testTon, y_predTon)
recallTon = recall_score(y_testTon, y_predTon)
f1scoreTon = f1_score(y_testTon, y_predTon)
tn, fp, fn, tp = confusion_matrix(y_testTon, y_predTon).ravel()
fpr_ton = fp / (fp + tn)

print("Metrics for Toniot dataset:")
print("Accuracy:", accuracyTon)
print("Precision:", precisionTon)
print("Recall:", recallTon)
print("F1-Score:", f1scoreTon)
print("FPR for Toniot dataset:", fpr_ton)



