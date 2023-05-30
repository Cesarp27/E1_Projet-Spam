# traitement des données
import pandas as pd

import matplotlib
matplotlib.use('TkAgg') # o 'Qt5Agg' # C'est juste pour voir les graphiques en Vscode

# Preprocessing
from sklearn.preprocessing import LabelEncoder 
from sklearn.feature_extraction.text import TfidfVectorizer

# Save models
import pickle

# Regression test
import unittest

data_news_sms = pd.read_csv('authentification/Projet_spam/SMSSpam2', sep='\t',header=None, names=['type', 'text'])


# data = pd.read_csv('spam.csv',encoding = "latin-1")
data = pd.read_csv('authentification/Projet_spam/spam.csv',encoding = "latin-1")

# Suppression of unnecessary columns
data = data.drop(['Unnamed: 2','Unnamed: 3','Unnamed: 4'], axis=1)
data = data.rename(columns={'v1': 'type', 'v2' : 'text'})

#Suppression of duplicate lines
data = data.drop_duplicates()

target = 'type'

# Récupération du modèle qui a donné les meilleurs résultats
# loaded_model_MNB = pickle.load(open('trained_model_MultinomialNB().pkl', 'rb'))

loaded_model_MNB = pickle.load(open(r'authentification\Projet_spam\trained_model_MultinomialNB().pkl', 'rb'))

def Pred_news_spams(df, target, model):

    # Cette étape sert à effectuer le processus inverse du "labelencoder" 
    # après que la prédiction a été effectuée
    lb_encod = LabelEncoder()
    y = lb_encod.fit_transform(data['type'])
    # ham = 0, spam = 1 

    #Traitement des messages avec TFIDF
    tfidf= TfidfVectorizer(max_features=3000)
    tfidf.fit(data['text'])

    # Dans ce cas, au lieu d'utiliser .fit_transform, nous utilisons simplement .transform 
    # et de cette manière, nous créons le TFIDF avec TfidfVectorizer qui a déjà été entraîné avec toutes les données.
    X_news=tfidf.transform(df['text']).toarray()

    # Nous faisons les prédictions avec notre modèle
    # ham = 0, spam = 1 
    predection = model.predict(X_news)

    # Nous faisons le processus inverse de label encoder 
    # pour trouver le nom de nos prédictions.
    predection_nom = lb_encod.inverse_transform(predection)

    # on trouve la probabilité de la prédiction
    proba = model.predict_proba(X_news)

    # nous ajoutons ces nouvelles informations à notre ensemble de données original
    df['prediction'] = predection_nom
    df['probabilite_spam'] = proba[:,1]*100
    df['probabilite_spam'] = round(df['probabilite_spam'], 1)

    return df

#data_news_sms_pred = Pred_news_spams(df=data_news_sms, target=target, model=loaded_model_MNB)

# print(data_news_sms_pred)