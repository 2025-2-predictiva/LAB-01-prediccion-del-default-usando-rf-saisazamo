# flake8: noqa: E501
#
# En este dataset se desea pronosticar el default (pago) del cliente el próximo
# mes a partir de 23 variables explicativas.
#
#   LIMIT_BAL: Monto del credito otorgado. Incluye el credito individual y el
#              credito familiar (suplementario).
#         SEX: Genero (1=male; 2=female).
#   EDUCATION: Educacion (0=N/A; 1=graduate school; 2=university; 3=high school; 4=others).
#    MARRIAGE: Estado civil (0=N/A; 1=married; 2=single; 3=others).
#         AGE: Edad (years).
#       PAY_0: Historia de pagos pasados. Estado del pago en septiembre, 2005.
#       PAY_2: Historia de pagos pasados. Estado del pago en agosto, 2005.
#       PAY_3: Historia de pagos pasados. Estado del pago en julio, 2005.
#       PAY_4: Historia de pagos pasados. Estado del pago en junio, 2005.
#       PAY_5: Historia de pagos pasados. Estado del pago en mayo, 2005.
#       PAY_6: Historia de pagos pasados. Estado del pago en abril, 2005.
#   BILL_AMT1: Historia de pagos pasados. Monto a pagar en septiembre, 2005.
#   BILL_AMT2: Historia de pagos pasados. Monto a pagar en agosto, 2005.
#   BILL_AMT3: Historia de pagos pasados. Monto a pagar en julio, 2005.
#   BILL_AMT4: Historia de pagos pasados. Monto a pagar en junio, 2005.
#   BILL_AMT5: Historia de pagos pasados. Monto a pagar en mayo, 2005.
#   BILL_AMT6: Historia de pagos pasados. Monto a pagar en abril, 2005.
#    PAY_AMT1: Historia de pagos pasados. Monto pagado en septiembre, 2005.
#    PAY_AMT2: Historia de pagos pasados. Monto pagado en agosto, 2005.
#    PAY_AMT3: Historia de pagos pasados. Monto pagado en julio, 2005.
#    PAY_AMT4: Historia de pagos pasados. Monto pagado en junio, 2005.
#    PAY_AMT5: Historia de pagos pasados. Monto pagado en mayo, 2005.
#    PAY_AMT6: Historia de pagos pasados. Monto pagado en abril, 2005.
#
# La variable "default payment next month" corresponde a la variable objetivo.
#
# El dataset ya se encuentra dividido en conjuntos de entrenamiento y prueba
# en la carpeta "files/input/".
#
# Los pasos que debe seguir para la construcción de un modelo de
# clasificación están descritos a continuación.
#
#
# Paso 1.
# Realice la limpieza de los datasets:
# - Renombre la columna "default payment next month" a "default".
# - Remueva la columna "ID".
# - Elimine los registros con informacion no disponible.
# - Para la columna EDUCATION, valores > 4 indican niveles superiores
#   de educación, agrupe estos valores en la categoría "others".
# - Renombre la columna "default payment next month" a "default"
# - Remueva la columna "ID".
#
#
# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test.
#
#
# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Ajusta un modelo de bosques aleatorios (rando forest).
#
#
# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use la función de precision
# balanceada para medir la precisión del modelo.
#
#
# Paso 5.
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.
#
#
# Paso 6.
# Calcule las metricas de precision, precision balanceada, recall,
# y f1-score para los conjuntos de entrenamiento y prueba.
# Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# Este diccionario tiene un campo para indicar si es el conjunto
# de entrenamiento o prueba. Por ejemplo:
#
# {'dataset': 'train', 'precision': 0.8, 'balanced_accuracy': 0.7, 'recall': 0.9, 'f1_score': 0.85}
# {'dataset': 'test', 'precision': 0.7, 'balanced_accuracy': 0.6, 'recall': 0.8, 'f1_score': 0.75}
#
#
# Paso 7.
# Calcule las matrices de confusion para los conjuntos de entrenamiento y
# prueba. Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'cm_matrix', 'dataset': 'train', 'true_0': {"predicted_0": 15562, "predicte_1": 666}, 'true_1': {"predicted_0": 3333, "predicted_1": 1444}}
# {'type': 'cm_matrix', 'dataset': 'test', 'true_0': {"predicted_0": 15562, "predicte_1": 650}, 'true_1': {"predicted_0": 2490, "predicted_1": 1420}}
#

import pandas as pd
import numpy as np
import json
import os
import gzip
import pickle
import joblib
import zipfile

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    balanced_accuracy_score, confusion_matrix
)


train = pd.read_csv(
    "files/input/train_data.csv.zip",  # archivo comprimido en .zip
    index_col=False,                      # no usar ninguna columna como índice
    compression="zip",                    # indicar que está comprimido
)

test = pd.read_csv(
    "files/input/test_data.csv.zip",  # archivo comprimido en .zip
    index_col=False,                      # no usar ninguna columna como índice
    compression="zip",                    # indicar que está comprimido
)


# Renombrar la variable target
train.rename(columns={"default payment next month": "default"}, inplace=True)
test.rename(columns={"default payment next month": "default"}, inplace=True)

# Eliminar ID
if "ID" in train.columns:
    train.drop(columns=["ID"], inplace=True)
    test.drop(columns=["ID"], inplace=True)
    

#Eliminar filas con NA
train.dropna(inplace=True)
test.dropna(inplace=True)


# EDUCATION > 4 => 4 (others)
train.loc[train["EDUCATION"] > 4, "EDUCATION"] = 4
test.loc[test["EDUCATION"] > 4, "EDUCATION"] = 4


# =======================
# Paso 2. División X/y
# =======================
X_train = train.drop(columns=["default"])
y_train = train["default"]

X_test = test.drop(columns=["default"])
y_test = test["default"]

# =======================
# Paso 3. Pipeline
# =======================
categorical_cols = ["SEX", "EDUCATION", "MARRIAGE",
                    "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]


preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
    ],
    remainder="passthrough"
)

rf = RandomForestClassifier(random_state=42, n_jobs=-1)

pipe = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", rf)
])

# =======================
# Paso 4. Optimización
# =======================
param_grid = {
    "model__n_estimators": [100, 300, 500],
    "model__max_depth": [10, 20, None],
    "model__min_samples_split": [2, 5]
}

grid = GridSearchCV(
    pipe,
    param_grid=param_grid,
    cv=10,
    scoring="balanced_accuracy",
    n_jobs=-1,
    verbose=2
)

grid.fit(X_train, y_train)


# =======================
# Paso 5. Guardar modelo
# =======================

best_model_path = 'files/models'
os.makedirs(best_model_path, exist_ok=True)

with gzip.open(os.path.join(best_model_path, "model.pkl.gz"), "wb") as f:
    pickle.dump(grid, f)
    

# =======================
# Paso 6. Métricas
# =======================
def compute_metrics(y_true, y_pred, dataset_name):
    return {
        "type": "metrics",
        "dataset": dataset_name,
        "precision": precision_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1_score": f1_score(y_true, y_pred)
    }

y_train_pred = grid.predict(X_train)
y_test_pred = grid.predict(X_test)

metrics = []
metrics.append(compute_metrics(y_train, y_train_pred, "train"))
metrics.append(compute_metrics(y_test, y_test_pred, "test"))

# =======================
# Paso 7. Confusion Matrix
# =======================
def compute_conf_matrix(y_true, y_pred, dataset_name):
    cm = confusion_matrix(y_true, y_pred, labels=[0,1])
    return {
        "type": "cm_matrix",
        "dataset": dataset_name,
        "true_0": {"predicted_0": int(cm[0,0]), "predicted_1": int(cm[0,1])},
        "true_1": {"predicted_0": int(cm[1,0]), "predicted_1": int(cm[1,1])}
    }

metrics.append(compute_conf_matrix(y_train, y_train_pred, "train"))
metrics.append(compute_conf_matrix(y_test, y_test_pred, "test"))

metrics_path = 'files/output/'
os.makedirs(metrics_path, exist_ok=True)

# Guardar JSON
with open("files/output/metrics.json", "w") as f:
    for m in metrics:
        f.write(json.dumps(m) + "\n")