# connexion avec la base de données
#from pymongo import MongoClient
import sqlite3

# traitement des données
import json
# import requests
import pandas as pd
import re
import os

# graphiques
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

import matplotlib
matplotlib.use('TkAgg') # o 'Qt5Agg' # Esto es solo para ver los graficos en Vscode


# nuage de mots
import numpy as np
from PIL import Image
# from wordcloud import WordCloud

# Preprocessing
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, RobustScaler, MinMaxScaler, FunctionTransformer


# import nltk
from sklearn.feature_extraction.text import TfidfVectorizer

# Pipeline and model
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import CategoricalNB, GaussianNB, BernoulliNB, MultinomialNB
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.tree import ExtraTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier 

# Score of models
#from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report, roc_curve, auc
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

# Save models
import pickle

# Regression test
import unittest


# parameters = paramètres à évaluer dans le GridSearcCV

# metric_GS = définir la métrique d'évaluation du modèle pour laquelle les hyperparamètres doivent être ajustés.

def entrainement_du_modele(data, modele_donne, parameters, metric_GS):
    # target preprocessing
    lb_encod = LabelEncoder()
    y = lb_encod.fit_transform(data['type'])
     
    
    tfidf= TfidfVectorizer(max_features=3000)
    
    X=tfidf.fit_transform(data['text']).toarray()
    
    # Division en groupes de training et d'évaluation
    from sklearn.model_selection import train_test_split
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=2, stratify=y)    
    

    # Declare model for Grid Search
    model_GS = modele_donne

    # Declare the pipeline
    pipe = Pipeline(steps=[
        # ('scaler', StandardScaler()),
        ('log_transform', FunctionTransformer(np.log1p, validate=True)), #scaler
        ('model', model_GS)]
        )
    
    metrics = ['accuracy', 'precision', 'recall', 'roc_auc', 'f1']
    
    # Declare the Grid Search method
    grid = GridSearchCV(estimator = pipe, param_grid = parameters, scoring = metrics,
                        refit = metric_GS, cv = 3, n_jobs =-1, verbose = 1)

    # Fit the model
    grid.fit(X_train, y_train)
    
    # save the model to disk
    filename = f'trained_model_{modele_donne}.pkl'
    pickle.dump(grid, open('authentification/Projet_spam/saved_models/'+filename, 'wb'))



data = pd.read_csv(r'authentification\Projet_spam\SMSSpamCollection', sep='\t',header=None, names=['type', 'text'])

metric_GS = 'roc_auc'

modele_donne = svm.SVC()
parameters = {'model__kernel':('linear', 'rbf'), 'model__C':[1, 10]}
entrainement_du_modele(data, modele_donne, parameters, metric_GS)

modele_donne = BernoulliNB()
parameters = {'model__alpha': [0.4, 0.5, 1.0], 'model__binarize': [0.0, 0.5, 1.0]}
entrainement_du_modele(data, modele_donne, parameters, metric_GS)

modele_donne = MultinomialNB()
parameters = {'model__alpha': [0.1, 0.5, 1.0, 2.0], 'model__fit_prior': [True, False]} 
entrainement_du_modele(data, modele_donne, parameters, metric_GS)

modele_donne = KNeighborsClassifier()
parameters = {'model__n_neighbors':[1,3,5,12,15], 'model__weights': ('uniform','distance')}
entrainement_du_modele(data, modele_donne, parameters, metric_GS)

modele_donne = GradientBoostingClassifier()
parameters = {'model__loss':('deviance', 'exponential'), 'model__learning_rate': [0.1, 0.2],
             'model__n_estimators':[100, 150]}
entrainement_du_modele(data, modele_donne, parameters, metric_GS)

modele_donne = LinearSVC()
parameters = {'model__C':[0.001,0.01,0.1], 'model__dual': [False,True],
             'model__multi_class':['ovr', 'crammer_singer']}
entrainement_du_modele(data, modele_donne, parameters, metric_GS)

modele_donne = AdaBoostClassifier()
parameters = {'model__n_estimators':[25, 50, 100], 'model__learning_rate': [0.5, 1.0, 2.0],
             'model__algorithm':['SAMME', 'SAMME.R']}
entrainement_du_modele(data, modele_donne, parameters, metric_GS)

modele_donne = RandomForestClassifier()
parameters = {'model__n_estimators':[50, 100, 150], 'model__criterion': ['gini', 'entropy'],
             'model__min_samples_split':[2, 5, 10]}
entrainement_du_modele(data, modele_donne, parameters, metric_GS)

modele_donne = ExtraTreesClassifier()
parameters = {'model__n_estimators':[50, 100, 150], 'model__criterion': ('gini', 'entropy'),
             'model__min_samples_split':[2, 3, 4]}
entrainement_du_modele(data, modele_donne, parameters, metric_GS)


