# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 17:35:13 2023

@author: nessi
"""

import subprocess
import numpy as np
import pandas as pd
import seaborn as sns
import shap
import requests
import json
import urllib.request
from lightgbm import LGBMClassifier
import pickle
#from PIL import Image

import plotly.express as px
import plotly.offline as py
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff

import streamlit as st
import streamlit.components.v1 as components

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.lines as mlines




def lancer_dash_glo():
    st.set_option('deprecation.showPyplotGlobalUse', False)

    #st.set_page_config(
    #    page_title= "Analyse globale ", 
    #    page_icon= "🧐🔎",
    #    layout="wide"
    #    )
    # Suppression des marges par défaut
    padding = 1
    st.markdown(f""" <style>
        .reportview-container .main .block-container{{
            padding-top: {padding}rem;
            padding-right: {padding}rem;
            padding-left: {padding}rem;
            padding-bottom: {padding}rem;
        }} </style> """, unsafe_allow_html=True)
    #Titre
    html_header="""
        <head> 
        <center>
            <title>Application Dashboard Crédit Score - Analyse Globale</title> <center>
            <meta charset="utf-8">
            <meta name="description" content="Analyse générale">
            <meta name="viewport" content="width=device-width, initial-scale=1">
        </head>             
        <h1 style="font-size:300%; color:Crimson; font-family:Arial"> Prêt à dépenser <br>
            <h2 style="color:Gray; font-family:Georgia"> Dashboard global</h2>
            <hr style= "  display: block;
              margin-top: 0;
              margin-bottom: 0;
              margin-left: auto;
              margin-right: auto;
              border-style: inset;
              border-width: 1.5px;"/>
         </h1>
    """
    st.markdown('<style>body{background-color: #fbfff0}</style>',unsafe_allow_html=True)
    st.markdown(html_header, unsafe_allow_html=True)

    #Chargement des données 
    best_model = pd.read_pickle('https://github.com/MohamedData/Projet_7/raw/main/best_model.pickle')
    test_origin = pd.read_pickle('https://github.com/MohamedData/Projet_7/raw/main/test_set.pickle') #pickle.load( open( "test_set", "rb" ) )
    test_native = pd.read_pickle('https://github.com/MohamedData/Projet_7/raw/main/test_prediction.pickle')

    #url=  "https://dash-scoring.herokuapp.com/prediction_complete"
    #with urllib.request.urlopen(url) as url:
    #    data = json.loads(url.read())
    #    #st.write(data)
    #    df =pd.DataFrame.from_dict(data)
    #df = df.T



    df = test_origin.reset_index()

    test_native = test_native.set_index('SK_ID_CURR').loc[df['SK_ID_CURR'].values].reset_index()


    st.write(test_native)

    st.sidebar.markdown("# 🎈 Analyse Globale ")


    st.sidebar.markdown("Présentation personnalisée pour chaque client, incluant leurs informations et leur score de crédit, expliqués par une vue d'ensemble. En parallèle, une analyse des variables cruciales et un tableau de bord interactif sont proposés.")


    liste_clients=list(test_native.SK_ID_CURR.values)

    ID_client = st.selectbox(
         "Merci de saisir l'identifiant du client:",
         (liste_clients))

    st.write("Exemple d'ID : 100172, 100013, 455955")

     #Récupération des informations du client

    data_client=test_native.loc[df['SK_ID_CURR']==int(ID_client), :]
    col1, col2 = st.columns(2)
    with col1:
        st.write('__Info crédit__')
        st.write('Type de contrat:', data_client['NAME_CONTRACT_TYPE'].values[0])
        st.write('Montant total du crédit:', data_client['AMT_CREDIT'].values[0],"$")
        st.write('Annuité:', data_client['AMT_ANNUITY'].values[0],"$")
        ratio_inc_cred = data_client['AMT_INCOME_TOTAL'].values[0] / data_client['AMT_CREDIT'].values[0]
        st.write('Ratio crédit sur revenus:', ratio_inc_cred,"$")
    with col2:
        st.write('__Info Client__')
        st.write('Genre:', data_client['CODE_GENDER'].values[0])
        st.write('Age :', round(abs(data_client['DAYS_BIRTH'].values[0]/365)))
        st.write('Status :', data_client['NAME_FAMILY_STATUS'].values[0])
        st.write('Education:', data_client['NAME_EDUCATION_TYPE'].values[0])
        st.write('Occupation:', data_client['OCCUPATION_TYPE'].values[0])
        st.write(
            'Employé:', data_client['ORGANIZATION_TYPE'].values[0],
            ', depuis', round(abs(data_client['DAYS_EMPLOYED'].values[0]/365)),
            'années')
        st.write("Type d'habitation:", data_client['NAME_HOUSING_TYPE'].values[0])
        st.write('Voiture :', data_client['FLAG_OWN_CAR'].values[0])
        st.write("Salaire annuel :",data_client.AMT_INCOME_TOTAL.values[0],"$")

        

    # Sidebar pour saisir l'ID client
    #st.sidebar.title("Saisir l'ID Client")
    #id_client = st.sidebar.number_input("ID Client", value=100001, min_value=100001, max_value=456255)

    # Bouton pour faire la prédiction
    if st.button("Analyse du risque"):
        # Faire une requête à l'API pour obtenir la prédiction
        url = f"https://api-hm-8e0d0e66dd33.herokuapp.com/predict_proba/{ID_client}"
        response = requests.get(url)
        prediction = response.json()

        # Déterminer si le crédit est accordé ou refusé
        if prediction['proba_classe_1'] < 0.28:
            st.success("Crédit accordé")
        else:
            st.error("Crédit refusé")
            
        st.write("Probabilité de défaut de paiement:", np.round(prediction['proba_classe_1'], 2))





    ## Récupération des features clients
    test = df.drop(columns =["SK_ID_CURR"])
    explainer_shap = shap.TreeExplainer(best_model)
    shap_value = explainer_shap.shap_values(test)
    #shap_id = df[df["SK_ID_CURR"]== ID_client].copy().T

    st.markdown("# ANALYSE GLOBALE ")

    fig,ax=plt.subplots( figsize=(10,4))
    ax = shap.summary_plot(shap_value, test)
    st.pyplot(fig)

    def application_samples_component():
        st.write('Sample size:')
        nb_clients_sample = st.number_input(
            label='Number of clients', 
            min_value=1,
            max_value=df.shape[0],
            format='%i')
        if st.button('Generate sample'):
            st.write(df.sample(nb_clients_sample))

    application_samples_component()
    # Recupération des indicateurs impliqués dans le calcul




    df_type=test_native.drop(columns =['SK_ID_CURR','PREDICTION'])
    df_num=df_type.select_dtypes(include = 'number')

    ## Select qualitative columns
    st.markdown("<h3 style='text-align: left; color: lightblue;'>Distribution des variables quantitatives</h3>", unsafe_allow_html=True)
    Col_num = st.selectbox(
         "Sélectionnez un indicateur pour une analyse analyse intéractive, le client se situe au repère rouge :",
         list(df_num.columns))

    st.subheader(Col_num)
    #fig,ax=plt.subplots( figsize=(10,4))
        
    x0 = test_native[test_native['PREDICTION']==0][Col_num]
    y0 = test_native[test_native['PREDICTION']==1][Col_num]
    z0 = test_native[Col_num]
    bins = np.linspace(0, 1, 15)

    risque_client=test_native[test_native['SK_ID_CURR']==ID_client][Col_num].item()
        

    group_labels = ['Solvable', 'Non solvable','Global']

    fig = go.Figure()
    fig.add_trace(go.Histogram(x=x0, name = 'Crédit accordé'))
    fig.add_trace(go.Histogram(x=y0, name = 'Risque de défaut'))
    fig.add_trace(go.Histogram(x=z0,name = 'Tous les cients' ))
    fig.add_vline(x= risque_client, annotation_text = 'client n° '+ str(ID_client), line_color = "red")
    fig.update_layout(barmode='relative')
    fig.update_traces(opacity=0.75)
    plt.show()
    st.plotly_chart(fig, use_container_width=True)

    df_object = df_type.select_dtypes(include = 'object')

    ## Select objects columns
    st.markdown("<h3 style='text-align: left; color: lightblue;'>Distribution des variables qualitatives</h3>", unsafe_allow_html=True)
    Col = st.selectbox(
         'Sélectionnez un indicateur :',
         list(df_object.columns))

    st.subheader(Col)
    sizes0 = list(test_native[Col][test_native['PREDICTION']==0].value_counts().values)
    labels0 =list(test_native[Col][test_native['PREDICTION']==0].value_counts().index)

    sizes1 = list(test_native[Col][test_native['PREDICTION']==1].value_counts().values)
    labels1 =list(test_native[Col][test_native['PREDICTION']==1].value_counts().index)

    size = list(test_native[Col].value_counts().values)
    labels = list(test_native[Col].value_counts().index)


    risque_client=test_native[test_native['SK_ID_CURR']== ID_client][Col].item()

    fig = make_subplots(rows=1, cols=3, specs=[[{'type':'domain'}, {'type':'domain'}, {'type':'domain'}]])

    fig.add_trace(go.Pie(labels=labels0, values = sizes0, name = 'Crédit accordé' ), 1,1)
    fig.add_trace(go.Pie(labels=labels1, values = sizes1, name ='Risque de défaut' ),1,2)
    fig.add_trace(go.Pie(labels=labels, values = size, name = 'Tous les cients'),1,3)
    #fig.add_vline(x= risque_client, annotation_text = 'client n° '+ str(id_client), line_color = "red")

    fig.update_traces(hole=.45, hoverinfo="label+percent+name")

    fig.update_layout(
        title_text="Répartition des clients",
        # Add annotations in the center of the donut pies.
        annotations=[dict(text='Solvable', x=0.1, y=0.5, font_size=10, showarrow=False),
                     dict(text='Non Solvable', x=0.5, y=0.5, font_size=10, showarrow=False),
                    dict(text='Global', x=0.89, y=0.5, font_size=10, showarrow=False)])
    st.plotly_chart(fig, use_container_width=False, sharing="streamlit",)

    ## Analyse dépendance des 3 features les plus importantes
    st.markdown("<h3 style='text-align: left; color: lightblue;'>Analyse des 3 indicateurs les plus importants </h3>", unsafe_allow_html=True)
    shap.dependence_plot("EXT_SOURCE_3", shap_value[0], test)
    st.pyplot()

    fig,ax=plt.subplots( figsize=(10,4))
    ax =shap.dependence_plot("CREDIT_TO_ANNUITY_RATIO", shap_value[0],test)
    st.pyplot()

    fig,ax=plt.subplots( figsize=(10,4))
    ax = shap.dependence_plot("EXT_SOURCE_2", shap_value[0], test)
    st.pyplot()

    ax =shap.dependence_plot("DAYS_BIRTH", shap_value[0],test)
    st.pyplot()

def lancer_dash_loc():
    #Options
    st.set_option('deprecation.showPyplotGlobalUse', False)

    #Nom de la page
    #st.set_page_config(
    #    page_title= "Analyse locale", 
    #    page_icon= "🧐🔎",
    #    layout="wide"
    #    )

    # Suppression des marges par défaut
    padding = 1
    st.markdown(f""" <style>
        .reportview-container .main .block-container{{
            padding-top: {padding}rem;
            padding-right: {padding}rem;
            padding-left: {padding}rem;
            padding-bottom: {padding}rem;
        }} </style> """, unsafe_allow_html=True)

    #Titre
    html_header="""
        <head> 
        <center>
            <title>Application Dashboard Crédit Score - Analyse Client</title> <center>
            <meta charset="utf-8">
            <meta name="description" content="Analyse client">
            <meta name="viewport" content="width=device-width, initial-scale=1">
        </head>             
        <h1 style="font-size:300%; color:Crimson; font-family:Arial"> Prêt à dépenser <br>
            <h2 style="color:Gray; font-family:Georgia"> Dashboard local</h2>
            <hr style= "  display: block;
              margin-top: 0;
              margin-bottom: 0;
              margin-left: auto;
              margin-right: auto;
              border-style: inset;
              border-width: 1.5px;"/>
         </h1>
    """
    st.markdown('<style>body{background-color: #fbfff0}</style>',unsafe_allow_html=True)
    st.markdown(html_header, unsafe_allow_html=True)
    #html_header

    #Chargement des données 
    best_model = pd.read_pickle('https://github.com/MohamedData/Projet_7/raw/main/best_model.pickle')
    test_origin = pd.read_pickle('https://github.com/MohamedData/Projet_7/raw/main/test_set.pickle') #pickle.load( open( "test_set", "rb" ) )
    test_native = pd.read_pickle('https://github.com/MohamedData/Projet_7/raw/main/test_prediction.pickle')

    ##url = "http://127.0.0.1:5000/prediction_complete"
    #url=  "https://dash-scoring.herokuapp.com/prediction_complete"
    #with urllib.request.urlopen(url) as url:
    #    data = json.loads(url.read())
    #    df =pd.DataFrame.from_dict(data)
    #df = df.T


    df = test_origin.reset_index()

    test_native = test_native.set_index('SK_ID_CURR').loc[df['SK_ID_CURR'].values].reset_index()


    st.write(test_native)

    st.sidebar.markdown("# 🎈 Analyse Locale ")


    st.sidebar.markdown("Analyse spécifique à un client, comprenant ses informations et son score de crédit, justifiés par une interprétation locale. De plus, des détails sur les profils similaires sont également inclus")


    html_select_client="""
        <div class="card">
          <div class="card-body" style="border-radius: 10px 10px 0px 0px;
                      background: #DEC7CB; padding-top: 5px; width: auto;
                      height: 40px;">
            <h3 class="card-title" style="background-color:#DEC7CB; color:Crimson;
                       font-family:Georgia; text-align: center; padding: 0px 0;">
              Informations sur le client / demande de prêt
            </h3>
          </div>
        </div>
        """

    #st.markdown(html_select_client, unsafe_allow_html=True)

    liste_clients=list(df.SK_ID_CURR.values)

    ID_client = st.selectbox(
         "Merci de saisir l'identifiant du client:",
         (liste_clients))

    st.write("Exemple d'ID : 205960, 413756, 344067")
     #Récupération des informations du client
    data_client=test_native[test_native.SK_ID_CURR==int(ID_client)]
    col1, col2 = st.columns(2)
    with col1:
        st.write('__Info crédit__')
        st.write('Type de contrat:', data_client['NAME_CONTRACT_TYPE'].values[0])
        st.write('Montant total du crédit:', data_client['AMT_CREDIT'].values[0],"$")
        st.write('Annuité:', data_client['AMT_ANNUITY'].values[0],"$")
        ratio_inc_cred = data_client['AMT_INCOME_TOTAL'].values[0] / data_client['AMT_CREDIT'].values[0]
        st.write('Ratio crédit sur revenus:', ratio_inc_cred,"$")
    with col2:
        st.write('__Info Client__')
        st.write('Genre:', data_client['CODE_GENDER'].values[0])
        st.write('Age :', round(abs(data_client['DAYS_BIRTH'].values[0]/365)))
        st.write('Status :', data_client['NAME_FAMILY_STATUS'].values[0])
        st.write('Education:', data_client['NAME_EDUCATION_TYPE'].values[0])
        st.write('Occupation:', data_client['OCCUPATION_TYPE'].values[0])
        st.write(
            'Employé:', data_client['ORGANIZATION_TYPE'].values[0],
            ', depuis', round(abs(data_client['DAYS_EMPLOYED'].values[0]/365)),
            'années')
        st.write("Type d'habitation:", data_client['NAME_HOUSING_TYPE'].values[0])
        st.write('Voiture :', data_client['FLAG_OWN_CAR'].values[0])
        st.write("Salaire annuel :",data_client.AMT_INCOME_TOTAL.values[0],"$")

    # Bouton pour faire la prédiction
    if st.button("Analyse du risque"):
        # Faire une requête à l'API pour obtenir la prédiction
        url = f"https://api-hm-8e0d0e66dd33.herokuapp.com/predict_proba/{ID_client}"
        response = requests.get(url)
        prediction = response.json()

        # Déterminer si le crédit est accordé ou refusé
        if prediction['proba_classe_1'] < 0.28:
            st.success("Crédit accordé")
        else:
            st.error("Crédit refusé")
        st.write("Probabilité de défaut de paiement:", np.round(prediction['proba_classe_1'], 2))
            
    ligne_pret = test_native[test_native.SK_ID_CURR == int(ID_client)][["Proba", "PREDICTION"]]
    score = round(ligne_pret.Proba.iloc[0]*100,2)

    #Tracé de la jauge
    st.markdown("<h3 style='text-align: left; color: lightblue;'>Score crédit</h3>", unsafe_allow_html=True)
    st.spinner('Jauge en cours de chargement')


    fig = go.Figure(go.Indicator(
        domain = {'x': [0, 1], 'y': [0, 1]},
        value = score,
        mode = "gauge+number",
        title = {'text': "Score crédit du client", 'font': {'size': 24}},
        delta = {'reference': 50},
        gauge = {'axis': {'range': [None, 100],
                          'tickwidth': 3,
                          'tickcolor': 'darkblue'},
                 'bar': {'color': 'white', 'thickness' : 0.15},
                 'bgcolor': 'white',
                 'borderwidth': 2,
                 'bordercolor': 'gray',
                 'steps': [{'range': [0, 40], 'color': 'Green'},
                           {'range': [40, 67.4], 'color': 'LimeGreen'},
                           {'range': [67.2, 67.8], 'color': 'red'},
                           {'range': [67.6, 80], 'color': 'Orange'},
                           {'range': [80, 100], 'color': 'Crimson'}],
                 'threshold': {'line': {'color': 'white', 'width': 5},
                               'thickness': 0.20,
                               # Score du client en %
                               # df_dashboard['SCORE_CLIENT_%']
                               'value': score }}))

    fig.update_layout(paper_bgcolor='white',
                            height=400, width=500,
                            font={'color': 'darkblue', 'family': 'Arial'},
                            margin=dict(l=0, r=0, b=0, t=0, pad=0))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("<h3 style='text-align: left; color: lightblue;'>Interprétabilité locale</h3>", unsafe_allow_html=True)

    #shap.initjs()

    test = df.drop(columns =["SK_ID_CURR"])
    #test_set = df.drop(columns =["Proba", "PREDICTION"])
    explainer = shap.TreeExplainer(best_model)
    shap_value = explainer(test, check_additivity=False)

    def affiche_facteurs_influence():
        ''' Affiche les facteurs d'influence du client courant
        '''
        html_facteurs_influence="""
            <div class="card">
                <div class="card-body" style="border-radius: 10px 10px 0px 0px;
                      background: #DEC7CB; padding-top: 5px; width: auto;
                      height: 40px;">
                      <h3 class="card-title" style="background-color:#DEC7CB; color:Crimson;
                          font-family:Georgia; text-align: center; padding: 0px 0;">
                          Variables importantes
                      </h3>
                </div>
            </div>
            """
        
        # ====================== GRAPHIQUES COMPARANT CLIENT COURANT / CLIENTS SIMILAIRES =========================== 
        
        if st.checkbox("Voir facteurs d\'influence"):     
            
            st.markdown(html_facteurs_influence, unsafe_allow_html=True)

            with st.spinner('**Affiche les facteurs d\'influence du client courant...**'):                 
                           
                #client_index = test_set[test_set['SK_ID_CURR'] == ID_client].index.item()
                #st.write(client_index)
                X_shap = df.set_index('SK_ID_CURR')
                #X_shap = X_shap.drop(columns = ["Proba","PREDICTION"])
                X_test_courant = X_shap.loc[ID_client]
                X_test_courant_array = X_test_courant.values.reshape(1, -1)
                    
                shap_values_courant = explainer.shap_values(X_test_courant_array)
                    
                    
                    # Forceplot du client courant
                    # BarPlot du client courant
                    
                st.pyplot(shap.plots.force(explainer.expected_value[1],shap_values_courant[1],X_test_courant, matplotlib=True))  
                   
                    
                        # Plot the graph on the dashboard
                    
         
                    # Décision plot du client courant
                        # Décision Plot
                shap.decision_plot(explainer.expected_value[1], shap_values_courant[1], X_test_courant)
                   
                        # Plot the graph on the dashboard
                st.pyplot()
    affiche_facteurs_influence()

    test_native = pd.read_pickle('https://github.com/MohamedData/Projet_7/raw/main/test_prediction.pickle')
    test_native['AGE']=round(abs(test_native['DAYS_BIRTH']/365),1)
      #ID_c=int(ID)



    #INFO CLIENT SHAP
    info_client=test_native[test_native['SK_ID_CURR']==ID_client]
    enfant_c=info_client['CNT_CHILDREN'].item()
    age_c=info_client['AGE'].item()
    genre_c=info_client['CODE_GENDER'].item()
    region_c=info_client['REGION_RATING_CLIENT'].item()
         
      #PROCHE VOISIN
    enfant_v=test_native[test_native['CNT_CHILDREN']==enfant_c]
    age_v=enfant_v[enfant_v['AGE']==age_c]
    genre_v=age_v[age_v['CODE_GENDER']==genre_c]
    region_v=genre_v[genre_v['REGION_RATING_CLIENT']==region_c]

    if len(region_v) < 15:
      shap_values=region_v.sample(len(region_v),random_state=42)
    if len(region_v) >= 15:
      shap_values=region_v.sample(15,random_state=42)

    fig,ax=plt.subplots( figsize=(10,4))
    plt.barh(range(len(shap_values)),shap_values['Proba'])
    risque_client=info_client['Proba'].item()
    plt.axhline(y=risque_client,linewidth=8, color='#d62728')
    plt.xlabel('% de risque')
    plt.ylabel('N° profils similaires')
    plt.figtext(0.755,0.855,'-',fontsize = 60,fontweight = 'bold',color = '#d62728')
    plt.figtext(0.797,0.9,'Client '+str(ID_client))
    st.pyplot(fig)

    moy_vois=shap_values['Proba'].mean()
    diff_proba=round(abs(risque_client-moy_vois)*100,2)
    st.write('Le client',str(ID_client),'à un écart de',str(diff_proba),'% de risque avec les clients de profils similaires.')
    
def afficher_page_accueil():
    #st.set_page_config(
    #    page_title = "Accueil",
    #    page_icon = "🔎",
    #    layout="wide"
    #    )
    


    # Suppression des marges par défaut
    padding = 1
    st.markdown(f""" <style>
        .reportview-container .main .block-container{{
            padding-top: {padding}rem;
            padding-right: {padding}rem;
            padding-left: {padding}rem;
            padding-bottom: {padding}rem;
        }} </style> """, unsafe_allow_html=True)
    #Titre
    html_header="""
        <head> 
        <center>
            <title>Dashboard Scoring client </title> <center>
            <meta charset="utf-8">
            <meta name="description" content="accueil">
            <meta name="viewport" content="width=device-width, initial-scale=1">
        </head>             
            <h1 style="color:Gray; font-family:Georgia"> Accueil </h1>
            <hr style= "  display: block;
              margin-top: 0;
              margin-bottom: 0;
              margin-left: auto;
              margin-right: auto;
              border-style: inset;
              border-width: 1.5px;"/>
         </h1>
    """
    #<h1 style="font-size:300%; color:Crimson; font-family:Arial"> Prêt à dépenser <br>
    #<title>Dashboard Scoring client </title> <center>
    st.markdown('<style>body{background-color: #fbfff0}</style>',unsafe_allow_html=True)
    st.markdown(html_header, unsafe_allow_html=True)
    st.markdown("<h3 style='font-size:18px; text-align:center;'>Tableau de bord descriptif présentant la décision d'octroi de crédit, suivi d'une explication des raisons sous un angle global et local qui ont conduit à cette décision.</h3>", unsafe_allow_html=True)
    #st.markdown("Accueil🔎")
    
    #st.sidebar.markdown(" Page d'accueil du dashboard. Ci-dessus le menu principal qui permet de naviguer dans les résultats notre modèle de scoring.")
    
    #st.markdown("#  <center>  Dashboard scoring client </center>", unsafe_allow_html=True)
    
    #st.sidebar.success("Selectionnez un mode d'analyse.")
    
#    logo =  Image.open("image\logo_entreprise.png") 
#    logo_home = Image.open("image\home_credit_logo.png")
    
    #col1, col2, col3 = st.columns([1,1,1])
    
   # with col1:
   #     st.write("")
    
    #with col2:
    #    st.image(logo, width= 600)
    
    #with col3:
    #    st.write("")
    
    
    #st.sidebar.image(logo_home, width=240, caption=" Origine des données ",
    #                 use_column_width='always')

def main():
    st.set_page_config(page_title="Mon Dashboard")
    
    #st.title("Mon Dashboard")
    afficher_page_accueil()
    # Créer un menu de navigation à gauche
    selection = st.selectbox("Sélectionnez une option", ["Sélectionnez une option", "Global", "Local"])
    if selection == "Global":
        lancer_dash_glo()        
    elif selection == "Local":
        lancer_dash_loc()
    


if __name__ == "__main__":
    main()

