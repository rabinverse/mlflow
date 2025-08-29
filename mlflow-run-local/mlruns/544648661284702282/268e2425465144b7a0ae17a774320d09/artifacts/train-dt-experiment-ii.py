import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)
import mlflow
import mlflow.sklearn


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


max_depth = 60


# model
dt = DecisionTreeClassifier(max_depth=max_depth)
dt.fit(x_train, y_train)
y_pred = dt.predict(x_test)

# accuracy=accuracy_score(y_test, y_pred)
# precision=precision_score(y_test, y_pred)
# recall_value=recall_score(y_test, y_pred)
# f1_score=f1_score(y_test, y_pred)

# just print
print("Accuracy : ", accuracy_score(y_test, y_pred))
print("Precision : ", precision_score(y_test, y_pred))
print("Recall Score : ", recall_score(y_test, y_pred))
print("f1_score : ", f1_score(y_test, y_pred))

# -------------> log experiment vars     <------------
# with mlflow.start_run(): without setexperimet or experiment id they are  stored in the default experiment runs

# mlflow.set_experiment(
#     "dt-Diabetes_classification"
# )  # - (new experiment name) if present it creates runs in that experiment if experiment doesnt exists  it will creat one
# or  with mlflow.start_run(experiment_id=the experiment id):
# deckare run name to avoid default random names run_name="name"
mlflow.set_tracking_uri("http://127.0.0.1:5000/")

mlflow.set_experiment("dt-Diabetes_classification")

with mlflow.start_run():
    mlflow.log_metric("Accuracy : ", float(accuracy_score(y_test, y_pred)))
    mlflow.log_metric("Precision : ", float(precision_score(y_test, y_pred)))
    mlflow.log_metric("Recall Score : ", float(recall_score(y_test, y_pred)))
    mlflow.log_metric("f1_score : ", float(f1_score(y_test, y_pred)))

    mlflow.log_param("max_depth", max_depth)

    mlflow.sklearn.log_model(
        sk_model=dt,
        artifact_path="DecisionTree",
        input_example=x_train[:5]  
    )

    # artifacts
    #  confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.title("Confusion Matrix")
    # Save the plot as an artifact
    plt.savefig("./data/confusion_matrix.png")
    mlflow.log_artifact("./data/confusion_matrix.png")
    mlflow.log_artifact(__file__)


# mlflow also saves code  as artifact
# mlflow.log_artifact(__file__)
