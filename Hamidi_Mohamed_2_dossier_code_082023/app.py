# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 21:22:49 2023

@author: nessi
"""

# Importation des librairies
from flask import Flask, request, jsonify
import pickle
import pandas as pd
from lightgbm import LGBMClassifier
import os

#Création d'une instance de Flask
app = Flask(__name__)

# Chargement du dataset de test
dataset = pd.read_pickle('https://github.com/MohamedData/Projet_7/raw/main/test_set.pickle')

# Chargement du modèle
model = pd.read_pickle('https://github.com/MohamedData/Projet_7/raw/main/best_model.pickle')

# Fonction de prédiction
def predict_proba(index):
    data = dataset.loc[index]
    proba = model.predict_proba([data])[0]
    return proba

# Récupération de l'index via l'URL
@app.route('/predict_proba/<int:index>', methods=['GET'])
# Fonction qui à un index renvoie les probabilités d'appartenance aux classes 0 et 1
def get_prediction_proba(index):
    proba = predict_proba(index)
    return jsonify({'proba_classe_0': float(proba[0]), 'proba_classe_1': float(proba[1])})

# Démarrage de l'application
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)