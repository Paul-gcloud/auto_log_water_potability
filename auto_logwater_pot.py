import pandas as pd
import numpy as np
import os
import pickle
from sklearn.model_selection import train_test_split
import mlflow
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import GradientBoostingClassifier
import mlflow.sklearn 
from sklearn.metrics import precision_score, accuracy_score, recall_score,f1_score, confusion_matrix

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("water_exp2")


data = pd.read_csv(r"C:\Users\USER\exp_mlflow\data\water_potability.csv")

#split data
train_data, test_data = train_test_split(data, test_size=0.20, random_state=42)


def fill_missing_with_median(df):
        for column in df.columns:
            if df[column].isnull().any():
                median_values=df[column].median()
                df[column]=df[column].fillna(median_values)
        return df
    
processed_train_data = fill_missing_with_median(train_data)
processed_test_data = fill_missing_with_median(test_data)

#prepare data
X_train = processed_train_data.drop(columns=['Potability'])
y_train = processed_train_data['Potability']

n_estimators = 500

with mlflow.start_run():
    #load_model
    clf=GradientBoostingClassifier(n_estimators=n_estimators)
    clf.fit(X_train, y_train)

    pickle.dump(clf,open('model.pkl', 'wb'))

    #load data
    #test_data = pd.read_csv('./data/processed/test_processed.csv')

    #prepare data
    X_test = processed_test_data.drop(columns=['Potability'])
    y_test = processed_test_data['Potability']

    #load model
    model = pickle.load(open('model.pkl', 'rb'))

    #prediction
    y_pred = model.predict(X_test)

    #Evaluation

    acc=accuracy_score(y_test, y_pred)
    precision=precision_score(y_test, y_pred)
    recall=recall_score(y_test, y_pred)
    f1score =f1_score(y_test, y_pred)

    mlflow.log_metric('acc',acc)
    mlflow.log_metric('precision',precision)
    mlflow.log_metric('recall',recall)
    mlflow.log_metric('flscore',f1score)

    mlflow.log_param('n_estimators', n_estimators)

    cm = confusion_matrix(y_test,y_pred)
    plt.figure(figsize=(5,5))
    sns.heatmap(cm, annot=True)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig('confusion_mat.png')

    mlflow.log_artifact('confusion_mat.png')

    mlflow.sklearn.log_model(sk_model=clf, name="GradientBoostingClassifier")

    mlflow.log_artifact(__file__)

    mlflow.set_tag("Data Scientst","Paul Mubiru")
    mlflow.set_tag("Model","GB")

    print('acc',acc)
    print('precision',precision)
    print('recall',recall)
    print('f1score',f1score)

        