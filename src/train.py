import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from dvclive.live import Live


data = pd.read_csv("./data/raw/diabetes.csv")

# ---------> filter
# print(data.head())
data["gender"] = data["gender"].map({"Female": 0, "Male": 1})
data.drop(columns="smoking_history", inplace=True)
# print(data.head())
data["gender"].fillna(data["gender"].mode()[0], inplace=True)
# print(data.isna().sum())
# ---------------->

X = data.drop(columns="diabetes")
y = data["diabetes"]

x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


n_estimators = 90
max_depth = 20


# model
rf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
rf.fit(x_train, y_train) 
y_pred = rf.predict(x_test)

# just print
print("Accuracy : ", accuracy_score(y_test, y_pred))
print("Precision : ", precision_score(y_test, y_pred))
print("Recall Score : ", recall_score(y_test, y_pred))
print("f1_score : ", f1_score(y_test, y_pred))

# -------------> log experiment vars     <------------
with Live(save_dvc_exp=True) as live:
    live.log_metric("Accuracy : ", float(accuracy_score(y_test, y_pred)))
    live.log_metric("Precision : ", float(precision_score(y_test, y_pred)))
    live.log_metric("Recall Score : ", float(recall_score(y_test, y_pred)))
    live.log_metric("f1_score : ", float(f1_score(y_test, y_pred)))

    live.log_param("n_estimator", n_estimators)
    live.log_param("max_depth", max_depth)