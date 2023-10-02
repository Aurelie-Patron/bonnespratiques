# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 15:08:58 2023

@author: AURELIE
"""

import streamlit as st
import random
@st.cache_data
def generate_random_value(x): 
  return random.uniform(0, x) 
a = generate_random_value(10) 
b = generate_random_value(20) 
st.write(a) 
st.write(b)


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_csv("train.csv")

st.title("Projet de classification binaire Titanic")
st.sidebar.title("Sommaire")
pages=["Exploration", "DataVizualization", "Modélisation"]
page=st.sidebar.radio("Aller vers", pages)

if page == pages[0] : 
  st.write("### Introduction")
  
  st.dataframe(df.head(10))
  st.write(df.shape)
  st.dataframe(df.describe())
   
  if st.checkbox ("Afficher les NA") :
      st.dataframe(df.isna().sum())

if page == pages[1] : 
  st.write("### DataVizualization") 

  fig = plt.figure()
  sns.countplot(x='Survived', data = df)
  st.pyplot(fig)
  
  fig = plt.figure()
  sns.countplot(x = 'Sex', data = df)
  plt.title("Répartition du genre des passagers")
  st.pyplot(fig)
  
  fig = plt.figure()
  sns.countplot(x = 'Pclass', data = df)
  plt.title("Répartition des classes des passagers")
  st.pyplot(fig)
  
  fig = sns.displot(x = 'Age', data = df)
  plt.title("Distribution de l'âge des passagers")
  st.pyplot(fig)

  fig = plt.figure()
  sns.countplot(x = 'Survived', hue='Sex', data = df)
  st.pyplot(fig)
  
  fig = sns.catplot(x='Pclass', y='Survived', data=df, kind='point')
  st.pyplot(fig)
  
  fig = sns.lmplot(x='Age', y='Survived', hue="Pclass", data=df)
  st.pyplot(fig) 

  # Supprimez les colonnes non numériques du DataFrame : elles ne peuvent pas être utilisée pour afficher une heatmap
  df_numeric = df.select_dtypes(include=['number'])
  # Affichez la carte de chaleur basée sur la corrélation des colonnes numériques

  fig, ax = plt.subplots()
  sns.heatmap(df_numeric.corr(), ax=ax)
  st.write(fig)     
  
if page == pages[2] : 
  st.write("### Modélisation")
  
  
  
  import joblib
  from sklearn.metrics import confusion_matrix

# Charger le modèle sauvegardé
  loaded_model = joblib.load("model")

# Les données d'entraînement (X_train et y_train) doivent également être disponibles ici
  df = df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)

  y = df['Survived']
  X_cat = df[['Pclass', 'Sex',  'Embarked']]
  X_num = df[['Age', 'Fare', 'SibSp', 'Parch']]

  for col in X_cat.columns:
    X_cat[col] = X_cat[col].fillna(X_cat[col].mode()[0])
    
  for col in X_num.columns:
    X_num[col] = X_num[col].fillna(X_num[col].median())
    
  X_cat_scaled = pd.get_dummies(X_cat, columns=X_cat.columns)

  X = pd.concat([X_cat_scaled, X_num], axis = 1)

  from sklearn.model_selection import train_test_split
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

  from sklearn.preprocessing import StandardScaler
  scaler = StandardScaler()
  X_train[X_num.columns] = scaler.fit_transform(X_train[X_num.columns])
  X_test[X_num.columns] = scaler.transform(X_test[X_num.columns])
  
# Interface utilisateur pour choisir le modèle
  selected_model = st.selectbox('Choix du modèle', ['Random Forest', 'SVC', 'Logistic Regression'])

  if selected_model:
    st.write('Le modèle choisi est :', selected_model)
    
    # Charger le modèle correspondant en fonction du choix de l'utilisateur
    if selected_model == 'Random Forest':
        model = joblib.load("Random Forest.pkl")  # Remplacez "random_forest_model.pkl" par le nom de fichier correct
    elif selected_model == 'SVC':
        model = joblib.load("SVC.pkl")  # Remplacez "svc_model.pkl" par le nom de fichier correct
    elif selected_model == 'Logistic Regression':
        model = joblib.load("logistic Regression.pkl")  # Remplacez "logistic_regression_model.pkl" par le nom de fichier correct

    # Afficher les scores en fonction du choix de l'utilisateur
    display = st.radio('Que souhaitez-vous montrer ?', ('Accuracy', 'Confusion matrix'))
    
    if display == 'Accuracy':
        # Calculer et afficher l'accuracy
        accuracy = model.score(X_test, y_test)  # Assurez-vous que X_test et y_test sont disponibles
        st.write(f'Accuracy du modèle {selected_model}: {accuracy}')
    
    elif display == 'Confusion matrix':
        # Calculer et afficher la matrice de confusion
        y_pred = model.predict(X_test)  # Assurez-vous que X_test est disponible
        cm = confusion_matrix(y_test, y_pred)  # Assurez-vous que y_test est disponible
        st.write(f'Matrice de confusion du modèle {selected_model}:')
        st.write(cm)

  
  