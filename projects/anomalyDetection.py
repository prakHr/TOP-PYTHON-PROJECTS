import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import seaborn as sns
from collections import defaultdict
import re
import matplotlib.pyplot as plt
import collections
from d3m import index
from d3m.metadata.base import ArgumentType
from d3m.metadata.pipeline import Pipeline, PrimitiveStep
from pyod.models.iforest import IForest
import tods
from sklearn.ensemble import IsolationForest 
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
def update(sender_type,df,sender_id_col,dict_oh):
    df[sender_type] = df[sender_id_col]
    df[sender_type] = df[sender_type].str.replace('\d+','')
    df[sender_type] = df[sender_type].str.replace('-','')
    ST_Count_dummies = pd.get_dummies(df[sender_type])
    for i in ST_Count_dummies:
        n = str(i)+ "_S"
        ST_Count_dummies[n] = ST_Count_dummies[i]
        del ST_Count_dummies[str(i)]
    df = pd.concat([df,ST_Count_dummies],axis = 1)
    df.drop([sender_type],axis = 1,inplace = True)
    df[sender_id_col] = df[sender_id_col].str.extract('(\d+)',expand = False)
    t = [col for col in ST_Count_dummies]
    dict_oh[sender_id_col] = t
    return df,dict_oh
def fraudDetection(fraud_df,time_step_col,bene_country_col,label_col,sender_id_col,bene_id_col,USD_amount_col,model=IsolationForest()):
    df = fraud_df.copy()
    df.drop([time_step_col],axis=1,inplace=True)
    df = df.dropna(axis=0, how='any', thresh=None, subset=None, inplace=False)
    df_new     = df.copy()
    df_anomaly = pd.DataFrame()
    df_bene                   = pd.DataFrame()
    df_bene[bene_country_col] = df[bene_country_col]
    df_new                    = df.loc[df[label_col]==1]
    x          = df_new[bene_country_col].value_counts()
    df_anomaly = pd.concat([df_anomaly,x],axis = 1)
    Anomaly_ratio = 'Anomaly_ratio'
    df_anomaly[Anomaly_ratio] = df_anomaly[bene_country_col]
    del df_anomaly[bene_country_col]
    y          = df[bene_country_col].value_counts()
    df_anomaly = pd.concat([df_anomaly,y],axis = 1)
    df_anomaly[Anomaly_ratio] = df_anomaly[Anomaly_ratio]/df_anomaly[bene_country_col]
    df_anomaly                = df_anomaly.dropna()
    df_anomaly                = df_anomaly.sort_values(by=Anomaly_ratio,ascending = False)
    sender_types = ['Sender_Type','Bene_Type']
    id_cols      = [sender_id_col,bene_id_col]
    dict_oh = {}
    for id_col,sender_type in zip(id_cols,sender_types):
        df,dict_oh = update(sender_type,df,id_col,dict_oh)
    df1   = df.copy()
    data  =  df.copy()
    data  = data.drop(columns = [label_col])
    data1 = df.copy()
    df_L  = data1[label_col]
    to_model_columns = data.columns[0:]
    data             = data[to_model_columns]
    X = np.expand_dims(data,axis = 1)
    X = np.squeeze(x)
    X = X.values.reshape(1, -1)
    y_true = np.expand_dims(df_L,axis = 1)
    y_true = np.squeeze(y_true)
    transformer_IF = model
    transformer_IF.fit(X)
    prediction_labels_IF = transformer_IF.predict(X)
    # prediction_score_IF  = transformer_IF.predict_score(X)
    y_pred = prediction_labels_IF
    df1['anomaly_IF'] = pd.Series(prediction_labels_IF.flatten())
    df1['anomaly_IF'] = df1['anomaly_IF'].apply(lambda x: x == 1)
    df1['anomaly_IF'] = df1['anomaly_IF'].astype(int)
    a  = df1.loc[df1['anomaly_IF']    == 1][:]
    a1 = a.loc[a[label_col]           == 1]
    a2 = df1.loc[df1[label_col]       == 1]
    ac = df_new.loc[df_new[label_col] == 1]
    predicted_anomaly           = a[USD_amount_col]
    correctly_predicted_anomaly = a1[USD_amount_col]
    return dict_oh,df,predicted_anomaly
if __name__=="__main__":
    path = "Fraud.csv"
    fraud_df = pd.read_csv(path)
    time_step_col    = "Time_step"
    bene_country_col = "Bene_Country"
    label_col        = "Label"
    sender_id_col    = "Sender_Id"
    bene_id_col      = "Bene_Id"
    USD_amount_col   = "USD_amount"
    rv = fraudDetection(fraud_df,time_step_col,bene_country_col,label_col,sender_id_col,bene_id_col,USD_amount_col)
    print(f"predicted_anomaly   = {rv[1]}")