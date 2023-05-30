# connexion avec la base de données
#from pymongo import MongoClient
# import sqlite3

# traitement des données
# import json
# import requests
import pandas as pd
# import re
# import os

# graphiques
# import matplotlib.pyplot as plt
# import seaborn as sns
import plotly.express as px

import matplotlib
matplotlib.use('TkAgg') # o 'Qt5Agg' # Esto es solo para ver los graficos en Vscode


# nuage de mots
# import numpy as np
# from PIL import Image
# from wordcloud import WordCloud

# Preprocessing
# from sklearn.impute import SimpleImputer
# from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder #, OneHotEncoder, StandardScaler, RobustScaler, MinMaxScaler, FunctionTransformer


# import nltk
from sklearn.feature_extraction.text import TfidfVectorizer

# Pipeline and model
# from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
# from sklearn.model_selection import GridSearchCV

# from sklearn.linear_model import LogisticRegression
# from sklearn.naive_bayes import CategoricalNB, GaussianNB, BernoulliNB, MultinomialNB
# from sklearn.ensemble import GradientBoostingClassifier
# from sklearn.ensemble import BaggingClassifier
# from sklearn.ensemble import ExtraTreesClassifier
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.ensemble import AdaBoostClassifier
# from sklearn.svm import SVC
# from sklearn.svm import LinearSVC
# from sklearn.tree import ExtraTreeClassifier
# from sklearn.naive_bayes import GaussianNB
# from sklearn import svm
# from sklearn.neighbors import KNeighborsClassifier 

# Score of models
#from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import classification_report, roc_curve, auc
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

# Save models
import pickle

# Regression test
import unittest


def evaluation_du_modele(data, modele_donne, modele_entraine):
    
    # # target preprocessing
    # lb_encod = LabelEncoder()
    # y = lb_encod.fit_transform(data['type'])
    # # ham = 0, spam = 1    
    
    # # features preprocessing
    # X = data.drop(columns='type')
    # X.head()    

    # # Division en groupes de training et d'évaluation
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=2, stratify=y)
    
    # target preprocessing
    lb_encod = LabelEncoder()
    y = lb_encod.fit_transform(data['type'])
     
    tfidf= TfidfVectorizer(max_features=3000)
    
    X=tfidf.fit_transform(data['text']).toarray()
    
    # Division en groupes de training et d'évaluation
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=2, stratify=y)    
    

    # # Declare model for Grid Search
    # model_GS = modele_donne

    # # Declare the pipeline
    # pipe = Pipeline(steps=[
    #     # ('scaler', StandardScaler()),
    #     ('log_transform', FunctionTransformer(np.log1p, validate=True)), #scaler
    #     ('model', model_GS)]
    #     )
    
    # metrics = ['accuracy', 'precision', 'recall', 'roc_auc', 'f1']
    
    # # Declare the Grid Search method
    # global grid
    # grid = GridSearchCV(estimator = pipe, param_grid = parameters, scoring = metrics,
    #                     refit = metric_GS, cv = 3, n_jobs =-1, verbose = 1)

    # # Fit the model
    # grid.fit(X_train, y_train)
    
    # # save the model to disk
    # filename = f'finalized_model_{modele_donne}.pkl'
    # pickle.dump(grid, open(filename, 'wb'))

    # Evaluate cross validation performance 
    # Nous utiliserons ici le GirdSearch déjà entraîné et stocké auparavant.
    
    # print()
    # print("model: ", modele_donne)
    # print("CV - Best score:", round(modele_entraine.best_score_,3))
    # print("CV - best parameters:", modele_entraine.best_params_)
        #print("CV - best estimator :", modele_entraine.best_params_)
    
    # cv_results_['mean_fit_time'] donne un array avec le résultat de chaque split, 
    # cette fonction fait une moyenne de toutes ces valeurs.
    def moyennes(keys_cv):        
        a1 = modele_entraine.cv_results_[keys_cv]
        Avg_key = sum(a1) / float(len(a1))
        #print(Avg_key)
        return Avg_key
    
    # Make predictions
    y_pred = modele_entraine.predict(X_test)
    
    # Evaluate model performance
    # print()    
    # print("++ CV - mean fit time:", round(moyennes('mean_fit_time'),2), 'seg', '++')
    time = round(moyennes('mean_fit_time'),2)
    # print()
        #print("CV - mean_test_accuracy:", round(moyennes('mean_test_accuracy'),3))
    # print("Test Accuracy:", round(accuracy_score(y_test, y_pred),3))
    accuracy_final = round(accuracy_score(y_test, y_pred),3)
    
    # print()
        #print("CV - mean_test_precision:", round(moyennes('mean_test_precision'),3))
    # print("Test precision:", round(precision_score(y_test, y_pred),3))
    precision_final = round(precision_score(y_test, y_pred),3)
    
    # print()
        #print("CV - mean_test_recall:", round(moyennes('mean_test_recall'),3))
    # print("Test recall:", round(recall_score(y_test, y_pred),3))
    recall_final = round(recall_score(y_test, y_pred),3)
    
    # print()
        #print("CV - mean_test_f1:", round(moyennes('mean_test_f1'),3))
    # print("Test f1:", round(f1_score(y_test, y_pred),3))
    f1_final = round(f1_score(y_test, y_pred),3)
    
    # print()
        #print("CV - mean_test_roc_auc:", round(moyennes('mean_test_roc_auc'),3))
    # print("Test roc_auc:", round(roc_auc_score(y_test, y_pred),3))
    roc_auc_final = round(roc_auc_score(y_test, y_pred),3)
        
    # print()
    # print("classification_report:")
    # print()
    # print(classification_report(y_test, y_pred))
    
    
    
    
    # # Matrice de confusion
    # cm=confusion_matrix(y_test,y_pred)
    # class_names = [0,1]
    # fig,ax = plt.subplots()
    # tick_marks = np.arange(len(class_names))
    # plt.xticks(tick_marks,class_names)
    # plt.yticks(tick_marks,class_names)
    # group_names = ['True Negative','False Positive','False Negative','True Positive']
    # group_counts = ['{0:0.0f}'.format(value) for value in
    #             cm.flatten()]
    # group_percentages = ['{0:.2%}'.format(value) for value in
    #                  cm.flatten()/np.sum(cm)]
    # labels = [f'{v1}\n{v2}\n{v3}' for v1, v2, v3 in
    #       zip(group_names,group_counts,group_percentages)]
    # labels = np.asarray(labels).reshape(2,2)
    # sns.heatmap(cm, annot=labels, fmt='',cmap="BuPu")
    # ax.xaxis.set_label_position('top')
    # plt.tight_layout()
    # plt.title('Confusion matrix')
    # plt.ylabel('Actual label')
    # plt.xlabel('Predicted label')
    # plt.show()
    
    
    # try:        
    #     # Make predictions and Courbe de ROC
    #     y_pred = modele_entraine.predict(X_test)
    #     global y_pred_proba
    #     y_pred_proba =modele_entraine.predict_proba(X_test)[:, 1]
    #     global fpr
    #     global tpr
    #     global thresholds
    #     fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)

    #     global fig_roc
    #     fig_roc = px.area(
    #     x=fpr, y=tpr,
    #     title=f'ROC Curve (AUC={auc(fpr, tpr):.4f})',
    #     labels=dict(x='False Positive Rate', y='True Positive Rate'),
    #     width=700, height=500)
    #     fig_roc.add_shape(
    #         type='line', line=dict(dash='dash'),
    #         x0=0, x1=1, y0=0, y1=1
    #     )

    #     fig_roc.update_yaxes(scaleanchor="x", scaleratio=1)
    #     fig_roc.update_xaxes(constrain='domain')       
     
    # except: 
    #     print("Cet estimateur n'a pas la propriété predict_proba pour pouvoir calculer la courbe ROC.")
    
    
    # try:        
    #     FI = modele_entraine.best_estimator_[1].feature_importances_
        
    #     d_feature = {'Stats':X.columns,
    #          'FI':FI}
    #     df_feature = pd.DataFrame(d_feature)

    #     df_feature = df_feature.sort_values(by='FI', ascending=0)
    #     print(df_feature)

    #     fig = px.bar_polar(df_feature, r="FI", theta="Stats",
    #                        color="Stats", template="plotly_dark",
    #                        color_discrete_sequence= px.colors.sequential.Plasma_r)
    #     fig.show()       
     
    # except:
    #     print()
    #     print('**********************************************************')
    #     print("Cet estimateur n'a pas la propriété de feature importances")
    #     print('**********************************************************')
        
       
    
    df_test_1 = pd.DataFrame(index=['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'time'])
    df_test_1[modele_donne] = (accuracy_final, precision_final, recall_final, f1_final, roc_auc_final, time)
    
    return df_test_1

## data = pd.read_csv(r'authentification\Projet_spam\SMSSpamCollection', sep='\t',header=None, names=['type', 'text'])

data = pd.read_csv('authentification/Projet_spam/spam.csv',encoding = "latin-1")
# Suppression of unnecessary columns
data = data.drop(['Unnamed: 2','Unnamed: 3','Unnamed: 4'], axis=1)
data = data.rename(columns={'v1': 'type', 'v2' : 'text'})

#Suppression of duplicate lines
data = data.drop_duplicates()


# SVC
loaded_model_SVC = pickle.load(open(r'authentification\Projet_spam\trained_model_SVC().pkl', 'rb'))
modele_donne = "svm.SVC"
df1 = evaluation_du_modele(data, modele_donne, modele_entraine = loaded_model_SVC)
# print(df)

# BernoulliNB
loaded_model_bnb = pickle.load(open(r'authentification\Projet_spam\trained_model_BernoulliNB().pkl', 'rb'))
modele_donne = "BernoulliNB"
df2 = evaluation_du_modele(data, modele_donne, modele_entraine = loaded_model_bnb)

# MultinomialNB
loaded_model_MNB = pickle.load(open(r'authentification\Projet_spam\trained_model_MultinomialNB().pkl', 'rb'))
modele_donne = "MultinomialNB"
df3 = evaluation_du_modele(data, modele_donne, modele_entraine = loaded_model_MNB)

# KNN
loaded_model_knn = pickle.load(open(r'authentification\Projet_spam\trained_model_KNeighborsClassifier().pkl', 'rb'))
modele_donne = "KNeighborsClassifier"
df4 = evaluation_du_modele(data, modele_donne, modele_entraine = loaded_model_knn)

# GradientBoosting
loaded_model_GBC = pickle.load(open(r'authentification\Projet_spam\trained_model_GradientBoostingClassifier().pkl', 'rb'))
modele_donne = "GradientBoostingClassifier"
df5 = evaluation_du_modele(data, modele_donne, modele_entraine = loaded_model_GBC)

# LinearSVC
loaded_model_LSVC = pickle.load(open(r'authentification\Projet_spam\trained_model_LinearSVC().pkl', 'rb'))
modele_donne = "LinearSVC"
df6 = evaluation_du_modele(data, modele_donne, modele_entraine = loaded_model_LSVC)

# AdaBoost
loaded_model_ABC = pickle.load(open(r'authentification\Projet_spam\trained_model_AdaBoostClassifier().pkl', 'rb'))
modele_donne = "AdaBoostClassifier"
df7 = evaluation_du_modele(data, modele_donne, modele_entraine = loaded_model_ABC)

# RandomForest
loaded_model_RF = pickle.load(open(r'authentification\Projet_spam\trained_model_RandomForestClassifier().pkl', 'rb'))
modele_donne = "RandomForestClassifier"
df8 = evaluation_du_modele(data, modele_donne, modele_entraine = loaded_model_RF)

# ExtraTrees
loaded_model_ETC = pickle.load(open(r'authentification\Projet_spam\trained_model_ExtraTreesClassifier().pkl', 'rb'))
modele_donne = "ExtraTreesClassifier"
df9 = evaluation_du_modele(data, modele_donne, modele_entraine = loaded_model_ETC)


df_tot_1 = pd.concat([df1, df2, df3, df4, df5, df6, df7, df8, df9], axis=1)
df_tot_1.columns = ['SVC', 'BernoulliNB', 'MultinomialNB', 'KNeighbors', 'GradientBoosting', 'LinearSVC', 'AdaBoost', 'RandomForest', 'ExtraTrees']
df_tot = df_tot_1.transpose()
# df_tot.sort_values(by = 'roc_auc', ascending= False, inplace= True)
df_tot.sort_values(by = 'accuracy', ascending= False, inplace= True)

df_tot_time = df_tot['time']
df_tot_heatmap = df_tot.drop(columns='time')

# df_tot_heatmap = df_tot_heatmap.reset_index()
# df_tot_heatmap = df_tot_heatmap.rename(columns={'index': 'model'})
# df_tot_heatmap.to_csv('authentification/Projet_spam/saved_models/metrics_heatmap.csv', index=True)
# print(df_tot_heatmap)


def plot_heatmap_spam(df):
    fig = px.imshow(df)
    fig.update_layout(xaxis={'title': 'Metrics'}, yaxis={'title': 'Models'}, width=500, height=550)
    # Ajouter des annotations
    annotations = []
    for i, row in enumerate(df.values):
        for j, value in enumerate(row):
            annotations.append(dict(text=str(value),
                                    x=j,
                                    y=i,
                                    font=dict(color='gray', size=12),
                                    showarrow=False))
    fig.update_layout(annotations=annotations,
                      title={'text': "Métriques de performance", 
                             'y':0.96, 
                             'x':0.5, 
                             'xanchor': 'center', 
                             'yanchor': 'top'}
                      )
    # fig.show()
    return fig
    
# plot_heatmap_spam(df_tot_heatmap)

def plot_time_heatmap(df):
    df = df.reset_index()
    df.columns = ['model', 'time']

    df = df.reset_index()
    df['order'] = df.index
    df_pivot = df.pivot(index='order', columns='time', values='time')
    df_pivot = df_pivot.set_index(df['model'])

    fig2 = px.imshow(df_pivot, labels=dict(x="Time (s)", y="Models"))
    fig2.update_layout(width=550, height=550, font=dict(size=12))
    fig2.update_layout(
        title={
            'text': "Temps moyen d'entraînement avec GridSearchCV (s)",  
            'y':0.96,  
            'x':0.5,  
            'xanchor': 'center',  
            'yanchor': 'top'  
        }
    )
    #fig2.show()
    return fig2

# fig3 = plot_time_heatmap(df = df_tot_time)
# fig3.show()