# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 18:44:41 2022

@author: Siddhartha-Sarkar
"""

    
 ###  Import Libreries  
import streamlit as st
from streamlit_option_menu import option_menu
st.set_option('deprecation.showPyplotGlobalUse', False)
import contractions 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image 
import warnings
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings(action='ignore', category=DataConversionWarning)
from collections import  Counter
import inflect
from bs4 import BeautifulSoup
import streamlit.components.v1 as components
import os
#for model-building
from sklearn.model_selection import train_test_split
import string
from tqdm import tqdm
#for model accuracy
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import plot_confusion_matrix
from sklearn.model_selection import cross_val_score
from numpy import mean
from numpy import absolute
from numpy import sqrt
#for visualization
import cufflinks as cf
cf.go_offline()
cf.set_config_file(offline=False, world_readable=True)
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.offline as offline
offline.init_notebook_mode()
from plotly import tools
import plotly.tools as tls
init_notebook_mode(connected=True)
import plotly.express as px
import plotly.graph_objects as go
import pickle
from joblib import dump, load
import joblib
# Utils
import time
timestr = time.strftime("%Y%m%d-%H%M%S")
import sys
from sklearn.metrics import r2_score,accuracy_score,mean_absolute_error,mean_squared_error,mean_absolute_percentage_error 
from sklearn.ensemble import ExtraTreesRegressor,RandomForestRegressor,AdaBoostRegressor,GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from plotly import tools
from sklearn.model_selection import train_test_split,cross_val_score,GridSearchCV,KFold
from sklearn.preprocessing import StandardScaler,LabelEncoder
import plotly.figure_factory as ff
import cufflinks as cf
from sklearn.model_selection import StratifiedKFold, cross_val_score, KFold
from sklearn.metrics import roc_auc_score
import matplotlib.gridspec as gridspec
# Setting seabon style
sns.set_style(style='darkgrid')
import scipy
import re
sns.set_palette("hls")
plt.style.use('fivethirtyeight')
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from scipy.stats.mstats import winsorize
from scipy.stats import boxcox, probplot, norm
from scipy.special import inv_boxcox
import random
import datetime
import math
le_encoder=LabelEncoder()
###############################################Data Processing###########################
data=pd.read_excel("Data_Train.xlsx")
loaded_model=pickle.load(open("Random_forest_intelligence.pkl","rb"))
# function for duration in minutes counting
def getDuration(x):
    replacements = [
    ('h', ':'),
    ('m', ''),
    (' ', '')]
    for old, new in replacements:
        x = re.sub(old, new, x)
    splt = x.split(':')
    hours_to_min = int(splt[0])*60
    if len(splt) == 2 and splt[1].isdigit():
        fin = hours_to_min + int(splt[1])
    else:
        fin = hours_to_min
    return fin

# function for duration counting     
def getDurationHours(x):
    replacements = [
    ('h', ':'),
    ('m', ''),
    (' ', '')]
    for old, new in replacements:
        x = re.sub(old, new, x)
    splt = x.split(':')
    hours_to_min = int(splt[0])*60
    if len(splt) == 2 and splt[1].isdigit():
        fin = hours_to_min + int(splt[1])
    else:
        fin = hours_to_min
    return splt[0]


def data_func(data):

    data['Duration_Minutes'] = data['Duration'].apply(getDuration) # duration of a flight
    data=data.drop(columns=["Date_of_Journey","Route",'Dep_Time','Arrival_Time'],axis=1)
    data["Price_wins"] = winsorize(data["Price"], limits = 0.01) 
    data["Duration_Minutes_wins"] = winsorize(data["Duration_Minutes"], limits = 0.01)
    data=data.drop(["Duration_Minutes_wins","Duration","Price"], axis=1)

    return data


dataframe=data_func(data)

def user_input_features():
    Date_of_Journey=st.sidebar.date_input("Enter your Journey Date") 
    #today=datetime.datetime.today()
    day=Date_of_Journey.day
    month=Date_of_Journey.month
    Airline = st.sidebar.selectbox("Airliner Coded name",(0,1,2,3,4,5,6,7,8,9,10,11))
    st.sidebar.markdown(""" 
                        ### Source City Name with coded values: Enter values from below list
                        
                        + Banglore   --->   [ 0 ]
                        + Chennai   --->    [ 1 ]
                        + Delhi     --->    [ 2 ]   
                        + Kolkata   --->    [ 3 ] 
                        + Mumbai    --->    [ 4 ]
                        """)
    Source = st.sidebar.selectbox("Source",(0,1,2,3,4))
    st.sidebar.markdown(""" 
                        ### Destination City Name with coded values: Enter values from below list
                          + Banglore    --->  [ 0 ]
                          + Cochin   --->     [ 1 ]    
                          + Delhi    --->     [ 2 ] 
                          + Hyderabad    ---> [ 3 ] 
                          + Kolkata      ---> [ 4 ] 
                          + New Delhi  --->   [ 5 ] 
                        """)
    Destination= st.sidebar.selectbox("Destination",(0,1,2,3,4,5))
    st.sidebar.markdown(""" 
                        ### No of Stops with coded values:  Enter values from below list
                            
                        + 1 stop   --->    [ 0  ] 
                        + 2 stops   --->   [ 1  ]
                        + 3 stops    --->  [ 2  ]    
                        + 4 stops     ---> [ 3  ]      
                        + non-stop   --->  [ 4  ]
                       """)

    Total_Stops = st.sidebar.selectbox("No of Stops",(0,1,2,3,4))
    st.sidebar.markdown(""" 
                        ### Additional Info with coded values: Enter values from below list
                            
                            
                        +  1 Long layover          --->          [ 0  ]
                        +  1 Short layover         --->          [ 1  ]
                        +  2 Long layover            --->        [ 2  ]
                        +  Business class          --->          [ 3  ]
                        +  Change airports         --->          [ 4  ]
                        +  In-flight meal not included   --->    [ 5  ]
                        +  No Info               --->            [ 6  ]
                        +  No check-in baggage included   --->   [ 7  ]
                        +  No info      --->                     [ 8  ]
                        +  Red-eye flight              --->      [ 9  ]
                          
                           """)
    Additional_Info = st.sidebar.selectbox("Additional_Info ",(0,1,2,3,4,5,6,7,8,9))
    
    Dep_Time = st.sidebar.time_input("Enter Departure Time ")
    Arrival_Time = st.sidebar.time_input(" Enter Arrival Time" ) 
    
    dep_hour=Dep_Time.hour
    dep_minute=Dep_Time.minute
    arr_hour=Arrival_Time.hour
    arr_minute=Arrival_Time.minute
    Duration_in_hour=abs(dep_hour-arr_hour)
    Duration_in_min=abs(dep_minute-arr_minute)
    Duration_Minutes=((Duration_in_hour*60)+Duration_in_min)
    #Duration_Minutes=st.sidebar.number_input("Enter The jouney time Duration in Minutes")
    
    
    
    data = {'day':day,
            'month':month,
            'Duration_Minutes':Duration_Minutes,
            'Airline':Airline,
            'Source':Source,
            'Destination':Destination,
            'Total_Stops':Total_Stops,
            'Additional_Info':Additional_Info,
            
            
            }

    features = pd.DataFrame(data,index = [0])
    
    return features
        


###############################################Exploratory Data Analysis###############################################

#For Label Analysis
def label_analysis():
    
    def plot1():
        dataframe=data_func(data)
        dataframe.groupby(by=["Source","Destination"])["Price_wins"].mean().plot(kind='bar',figsize=(16,6))
    p1=plot1()
    st.write("Distribution of  ticket Price")
    st.pyplot(p1)
    
    def plot2():
        dataframe=data_func(data)
        plt.figure(figsize=(15,8))
        plt.subplot(1,2,1)
        dataframe['Source'].value_counts().plot.pie(autopct='%1.1f%%')
        centre=plt.Circle((0,0),0.7,fc='white')
        fig=plt.gcf()
        fig.gca().add_artist(centre)
        plt.subplot(1,2,2)
        sns.countplot(x='Source',data=dataframe)
        plt.show()
    
    
    p2=plot2()
    st.write("Source Cities")
    st.pyplot(p2)
    def plot3():
        dataframe=data_func(data)
        plt.figure(figsize=(15,8))
        plt.subplot(1,2,1)
        dataframe['Destination'].value_counts().plot.pie(autopct='%1.1f%%')
        centre=plt.Circle((0,0),0.7,fc='white')
        fig=plt.gcf()
        fig.gca().add_artist(centre)
        plt.subplot(1,2,2)
        sns.countplot(x='Destination',data=dataframe)
        plt.show()
    
    
    p3=plot3()
    st.write("Destination cities")
    st.pyplot(p3)
    
    def plot4():
        dataframe=data_func(data)
        plt.figure(figsize=(15,8))
        plt.subplot(1,2,1)
        dataframe['Total_Stops'].value_counts().plot.pie(autopct='%1.1f%%')
        centre=plt.Circle((0,0),0.7,fc='white')
        fig=plt.gcf()
        fig.gca().add_artist(centre)
        plt.subplot(1,2,2)
        sns.countplot(x='Total_Stops',data=dataframe)
        plt.show()
    p4=plot4()
    st.write(" Distribution of Total Stops")
    st.pyplot(p4)
    
    def plot5():
        dataframe=data_func(data)
        plt.figure(figsize=(15,6))
        sns.countplot(dataframe['Source'],hue='Total_Stops',data=dataframe)
        plt.legend(loc='upper right')
        
    p5=plot5()
    st.write(" distributions of stops with source")
    st.pyplot(p5)    
    
    
    def plot7():
        dataframe=data_func(data)
        plt.figure(figsize=(15,6))
        sns.countplot(dataframe['Destination'],hue='Airline',data=dataframe,palette='Dark2')
        plt.legend(loc='upper right')
        
    p7=plot7()
    st.write("Airlines distributions with Destinations")
    st.pyplot(p7)  


  


def get_data_reg(dataframe):
    dataframe=data_func(data)
    dataframe['Source']=le_encoder.fit_transform(dataframe['Source'])
    dataframe["Destination"]=le_encoder.fit_transform(dataframe['Destination'])
    dataframe["Airline"]=le_encoder.fit_transform(dataframe['Airline'])
    dataframe["Additional_Info"]=le_encoder.fit_transform(dataframe['Additional_Info'])
    dataframe["Total_Stops"]=le_encoder.fit_transform(dataframe['Total_Stops'])
    
    X=dataframe.drop(columns=["Price_wins"],axis=1)
    y=dataframe["Price_wins"]
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42)
    
    return X_train,X_test,y_train,y_test  
    

############################################### Model Learning ###############################################
   
#Model Random Forest Regressor
def randomforest_regressor(dataframe):
    X_train,X_test,y_train,y_test=get_data_reg(dataframe)
    rf_reg=RandomForestRegressor(n_estimators=300,criterion='squared_error',max_depth=6,
                                 max_features='auto')
    rf_reg.fit(X_train,y_train)
    y_train_pred=rf_reg.predict(X_train)
    y_test_pred=rf_reg.predict(X_test)
    r2_score(y_train,y_train_pred)
    
    r2_score(y_test,y_test_pred)
    
    m1=mean_absolute_error(y_train,y_train_pred)
    st.write("Mean Absolute error: %0.3f " % m1)

    m2=mean_squared_error(y_train,y_train_pred)
    st.write("Mean squared error: %0.3f " % m2)
    m3=mean_absolute_percentage_error(y_train,y_train_pred)
    st.write("Mean absolute_percentage_error: %0.3f " % m3)
#Model XGBoost  Regressor
def XGboost_regressor(dataframe):
    X_train,X_test,y_train,y_test=get_data_reg(dataframe)
    xgb_reg=XGBRegressor(n_estimators=150,max_depth=6,learning_rate=0.05,booster="gbtree")
    xgb_reg.fit(X_train,y_train)
    y_train_pred=xgb_reg.predict(X_train)
    y_test_pred=xgb_reg.predict(X_test)
    r2_score(y_train,y_train_pred)
    
    r2_score(y_test,y_test_pred)
    
    m1=mean_absolute_error(y_train,y_train_pred)
    st.write("Mean Absolute error: %0.3f " % m1)

    m2=mean_squared_error(y_train,y_train_pred)
    st.write("Mean squared error: %0.3f " % m2)
    m3=mean_absolute_percentage_error(y_train,y_train_pred)
    st.write("Mean absolute_percentage_error: %0.3f " % m3)




# =============================================================================
# def predict_func(df):
#     df=user_input_features()
#     dataframe=data_func(data)
#     df["Source"]=le_encoder.fit_transform(df['Source'])
#     df["Destination"]=le_encoder.fit_transform(df['Destination'])
#     df["Airline"]=le_encoder.fit_transform(df['Airline'])
#     df["Additional_Info"]=le_encoder.fit_transform(df['Additional_Info'])
#     df["Total_Stops"]=le_encoder.fit_transform(df['Total_Stops'])
# # =============================================================================
# #     X_train,X_test,y_train,y_test=get_data_reg(dataframe)
# #     xgb_reg=XGBRegressor(n_estimators=150,max_depth=8,learning_rate=0.05,booster="gbtree")
# #     xgb_reg.fit(X_train,y_train)
# #     y_train_pred=xgb_reg.predict(X_train)
# # =============================================================================
#     y_test_pred=loaded_model.predict(df)
#     #xgb_prediction = xgb_classifer.predict(df)
#     return y_test_pred
# =============================================================================
###############################################Streamlit Main###############################################

def main():
    # set page title
    
    
            
    # 2. horizontal menu with custom style
    selected = option_menu(menu_title=None, options=["Home", "Projects", "About"], icons=["house", "book", "envelope"],  menu_icon="cast", default_index=0, orientation="horizontal", styles={"container": {"padding": "0!important", "background-color": " #f08080 "},"icon": {"color": "blue", "font-size": "25px"}, "nav-link": {"font-size": "25px","text-align": "left","margin": "0px","--hover-color": "#eee", },           "nav-link-selected": {"background-color": "green"},},)
    
    #horizontal Home selected
    if selected == "Home":
        st.title(f"You have selected {selected}")
        
        image= Image.open("airport_home.jpg")
        st.image(image,use_column_width=True)
            
        st.sidebar.title("Home")        
        with st.sidebar:
            image= Image.open("Home.png")
            add_image=st.image(image,use_column_width=True)  
        st.sidebar.markdown("[ Visit To Github Repositories](.git)")    
        st.balloons()
        st.title('Flight Fare Price Prediction')
        #st.video("https://youtu.be/O73OPzkUlR0")
        
        st.markdown("""

            This is a Flight Fare Price Prediction  Project using Machine Learning.
             
            #### Why  Flight Ticket price Fluctuates:
                
                Do fluctuating flight ticket prices ever perplex you? 
                One moment you find the flight rates to be exorbitant and within a week's time, 
                the prices drop down to your convenience? Has it ever left you wondering what could be 
                the right time to make a flight booking and get the best deal? Well, 
                it would be quite convenient if you knew when would be the perfect time 
                to snag those exciting deals and discounts on flight ticket price.
                
            #### The Main Reason For These Flactuations:
                
                + Distance-->Longer Distances Lead to Higher Airline Fares
                + Ticket Prices Change With the Season
                + Flight timing --> The Timing of the Flight Affects the Ticket Price
                + Flight travel type --> Business Travelers Affect Ticket Prices
                + Competition with other players
                + Airlines Price Tickets Based on Oil Prices
                + Airline Tickets Follow the Laws of Demand
            
           #### Using Machine Learning it is actually possible to reduce the uncertainty of flight ticket prices. So here i will be predicting the flight ticket prices using efficient machine learning techniques    
            
            #### About the dataset:

            + Airline: So this column will have all the types of airlines like Indigo, Jet Airways, Air India, and many more.
            + Date_of_Journey: This column will let us know about the date on which the passenger’s journey will start.
            + Source: This column holds the name of the place from where the passenger’s journey will start.
            + Destination: This column holds the name of the place to where passengers wanted to travel.
            + Route: Here we can know about that what is the route through which passengers have opted to travel from his/her source to their destination.
            + Arrival_Time: Arrival time is when the passenger will reach his/her destination.
            + Duration: Duration is the whole period that a flight will take to complete its journey from source to destination.
            + Total_Stops: This will let us know in how many places flights will stop there for the flight in the whole journey.
            + Additional_Info: In this column, we will get information about food, kind of food, and other amenities.
            + Price: Price of the flight for a complete journey including all the expenses before onboarding.
           
            
           
            """)
        ### features
        image= Image.open("word-image-20.png")
        st.image(image,use_column_width=True)
        st.header('Features')

        st.markdown("""
                #### Basic  Tasks:
                + App covers the most basic Machine Learning task of  Analysis, Correlation between variables,project report.
                
                
                
                #### Machine Learning:
                + Machine Learning on different Machine Algorithms, building different models and lastly  prediction. 
                
                """)
                
    #Horizontal About selected
    if selected == "About":
        st.title(f"You have selected {selected}")
        
        st.sidebar.title("About")
        with st.sidebar:
            image= Image.open("About_us.png")
            add_image=st.image(image,use_column_width=True)        
        
        #st.image('iidt_logo_137.png',use_column_width=True)
        st.markdown("<h2 style='text-align: center;'> This  is a Flight Fare Price Prediction Project :</h2>", unsafe_allow_html=True)

        st.markdown("""
                    #### @Author  Mr. Siddhartha Sarkar)
        
                    """)
        st.snow()
        image2= Image.open("Domestic-Flights_about.jpg")
        st.image(image2,use_column_width=True)
        st.markdown("[ Visit To Github Repositories](.git)")
    
    #Horizontal Project selected
    if selected == "Projects":
            st.title(f"You have selected {selected}")
            with st.sidebar:
                image= Image.open("pngwing.png")
                add_image=st.image(image,use_column_width=True) 
            import time

                
                          
            image2= Image.open("project_home_img.jpg")
            st.image(image2,use_column_width=True)
            my_bar = st.progress(0)
            for percent_complete in range(100):
                time.sleep(0.01)
                my_bar.progress(percent_complete + 1)
            st.sidebar.title("Navigation")
            menu_list1 = ['Exploratory Data Analysis',"Prediction With Machine Learning"]
            menu_Pre_Exp = st.sidebar.radio("Menu For Prediction & Exploratoriy", menu_list1)
            
            #EDA On Document File
            if menu_Pre_Exp == 'Exploratory Data Analysis' and selected == "Projects":
                    st.title('Exploratory Data Analysis')

                    
                    
                    menu_list2 = ['None', 'Analysis','Project_Report']
                    menu_Exp = st.sidebar.radio("Menu EDA", menu_list2)

                    
                    if menu_Exp == 'None':
                        st.markdown("""
                                    #### Kindly select from left Menu.
                                   # """)
                    
                    elif menu_Exp == 'Analysis':
                        label_analysis()

                    elif menu_Exp == 'Project_Report':
                        import dtale
                        d=dtale.show(data)
                        st.write("It is opening a New Browser Tabs ")
                        d.open_browser()


            elif menu_Pre_Exp == "Prediction With Machine Learning" and selected == "Projects":
                    st.title('Prediction With Machine Learning')
                    
                    menu_list3 = ['Checking ML Method And Accuracy' ,'Checking Regression Method And Accuracy' ,'Prediction' ]
                    menu_Pre = st.radio("Menu Prediction", menu_list3)
                    
                    #Checking ML Method And Accuracy
                    if menu_Pre == 'Checking ML Method And Accuracy':
                            st.title('Checking Accuracy On Different Algorithms')
                            dataframe=data_func(data)
                            
                            if st.checkbox("View data"):
                                st.write(dataframe)
                            #model = st.selectbox("ML Method",['Logistic Regression', 'XGB Classifier', 'Random Forest Classifier'])
                            #vector= st.selectbox("Vector Method",[ 'CountVectorizer' , 'TF-IDF'])

                            #if st.button('Analyze'):
                                #Logistic Regression 
                                #if model=='Logistic Regression':
                                    #logistic_regression(get_data_class(final_data))
                                    
                                

                                #XGB Classifier & CountVectorizer
                                #elif model=='XGB Classifier':
                                    #xgb_classifier(get_data_class(final_data))                                    
                                
                               
                                
                                
                                #Random Forest Classifier & CountVectorizer
                                #elif model=='Random Forest Classifier':
                                    #randomforest_classifier(get_data_class(final_data))
                              
                    #Checking ML Method And Accuracy
                    elif menu_Pre == 'Checking Regression Method And Accuracy':
                            st.title('Checking Accuracy On Different Algorithms')
                            dataframe=data_func(data)
                            
                            if st.checkbox("View data"):
                                st.write(dataframe)
                            model = st.selectbox("ML Method",['XGboost_regressor', 'Random Forest Regressor'])
                            #vector= st.selectbox("Vector Method",[ 'CountVectorizer' , 'TF-IDF'])

                            if st.button('Analyze'):
                                #Logistic Regression 
                                if model=='XGboost_regressor':
                                    XGboost_regressor(get_data_reg(dataframe))
                                    
                                

                                #XGB Classifier & CountVectorizer
                                elif model=='Random Forest Regressor':
                                    randomforest_regressor(get_data_reg(dataframe))                                    
                                
                              
                                          
                    elif menu_Pre == 'Prediction':
                        st.title('Prediction')
                        with st.spinner('Wait for it...'):
                            time.sleep(5)
    
                           
                        df= user_input_features()
                        
                        result_pred = int(loaded_model.predict(df))
                        
                        
                        st.markdown("""
                                ##### AirLiner Name with coded values:
                                +  Air Asia--->[ 0 ]
                                +  Air India--->[ 1 ]
                                +  GoAir --->[ 2 ]
                                +  IndiGo--->[ 3 ]
                                +  Jet Airways---> [ 4 ]
                                +  Jet Airways Business--->[ 5 ]
                                +  Multiple carriers--->[ 6 ]
                                +  Multiple carriers Premium economy--->[ 7 ]
                                +  SpiceJet--->[ 8 ]
                                +  Trujet--->[ 9 ] 
                                +  Vistara --->[ 10 ]
                                +  Vistara Premium economy--->[ 11 ] 
                                    
                                """)
                        st.success('Done!')       
                        st.success('The Flight Fare of Your Journey is--> {}'.format(result_pred))
                        
                            
                                

                                                      
if __name__=='__main__':
    main()            
            
            

