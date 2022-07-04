# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 11:51:52 2019

@author: BSI80086
"""

import streamlit as st
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import time
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

rawdata=pd.read_csv("emp_attrition.csv")
data = pd.read_csv("Employee Attrition.csv")
data_numeric= data.drop(['Department','EducationField','Gender','JobRole','MaritalStatus'],axis = 1)
data_numeric_short=data_numeric.drop(['BusinessTravel','attrition encode','gender encode','marital status encode','Human Resources',
                                      'Life Sciences','Marketing','Medical','Other','Technical Degree'],axis = 1)
dataset=pd.DataFrame({'Age':[0.0],'JobLevel':[0.0],'MonthlyIncome':[0.0],'NumCompaniesWorked':[0.0],'StockOptionLevel':[0.0],'TotalWorkingYears':[0.0],'Business Travel encode':[0.0]})

def main():
    st.sidebar.header('Welcome to Attrition Analysis, Visualization and Prediction!')
    page = st.sidebar.selectbox("What do you want?", ["Dataset","Dashboard","Prediction"], index=0,key=0)
    if page == "Dataset":
        st.image('attrition image.png')
        st.title("Dataset of Attrition")
        st.header("Here you can see both raw dataset and after pre-processing")
        if st.checkbox("Showing raw dataset?"):
            show_rawdata=st.slider("How much raw data do you want to see?" , 1,100,5)
            st.write(rawdata.sample(show_rawdata))
        if st.checkbox("Showing dataset after pre-processing?"):
            show_data=st.slider("How much data do you want to see?" , 1,100,5)
            st.write(data.sample(show_data))
        if st.checkbox("I want to check whether our data is balance or not? How is the comparison between resign and not resign employee?"):
            st.write(pd.DataFrame(data['Attrition'].value_counts()))
            proportion = data.Attrition.value_counts()[0] / data.Attrition.value_counts()[1]
            st.write("So, the data still can be tolerated since the proportion is %.2f" %proportion , ": 1")
            
    elif page == "Dashboard":
        st.image('HR-Analytics.jpg')
        st.title("Dashboard of Attrition Data")
        st.header("Here you can see data visualization and my analysis about each data in order to understand why I did pre-processing like that")
        data_type = st.sidebar.selectbox("What type of data you want to visualize and analyze?", ["Numerical","Categorical"])
        if data_type == "Numerical":
            if st.checkbox("Wanna know the how I analyze the data and the conclusion from my analysis?"):
                st.write("I use these steps to analyze how much each numerical factor affect attrition, : ")
                st.write("1. I sort the tables based on value from column I want to analyze ")
                st.write("2. I pick only 10% of the top/bottom data (depend on what column I analyze) or data ranking 1 to 147 or ranking 1323 to 1470 ")
                st.write("3. I count how many employee who resign from that 10% of data")
                st.write("4. I divide number I got from 3rd process with 147 in order to get percentage")
                st.write("5. If the percentage I got from 4th process is low, then the correlation between those column and attrition is also low")
                feature = st.selectbox('Choose category : ',data_numeric_short.columns)
                analytics(feature)
                
                
            if st.checkbox("Wanna see numerical data visualization?"):
                show_columns_x= st.selectbox("Select your x axis" , data_numeric.columns)
                show_columns_y= st.selectbox("Select your y axis", data_numeric.columns)
                show_color=st.selectbox("What data do you want as color?",data_numeric.columns)
                if st.button("Let's go!") :
                    with st.spinner('Wait a sec'):
                        time.sleep(2)
                        st.success('Here it is!')
                        fig = px.scatter(data,x= show_columns_x, y=show_columns_y, color= show_color, title='Distribution of attrition data')
                        st.plotly_chart(fig)
        
        if data_type == "Categorical":
            subject = st.radio('Choose category : ',['Department','EducationField',
                                                    'Gender','JobRole',
                                                    'MaritalStatus'])
            
            if st.button('Visualize!'):
                with st.spinner('Wait a sec'):
                    time.sleep(2)
                    st.success('Here it is!')
                    viz1(subject)
            
            
    elif page == "Prediction":
        st.image('prediction.jpg')
        st.title("Attrition Prediction")
        st.header('Every human capital could use this website to predict whether an employee will resign or not')
        
        name = st.text_input('Write employee name that you want to predict if they want to resign or not :')
        
        age=st.slider('Age :',17,61,18)
        dataset.Age[0]=age
        
        joblevel=st.slider ('What is his/her job level?', 1,6,2)
        dataset.JobLevel[0]=joblevel
        
        income = st.slider('How much his/her income?' ,1000,20000,1001)
        dataset.MonthlyIncome[0]=income
        
        numcompaniesworked=st.slider('How many company has he/she been working before?' , 0,10,1)
        dataset.NumCompaniesWorked[0]=numcompaniesworked
        
        stock=st.slider ('How many stock does he/she has in this company?',0,4,1)
        dataset.StockOptionLevel[0]=stock
       
        total_working_years=st.slider('How long he/she has been working? ',0,50,1)
        dataset.TotalWorkingYears[0]=total_working_years
        
        bistrip= st.radio('How often does he/she travel for business reason? ',['Never','Rarely','Often'])
        if bistrip=='Never':
            dataset['Business Travel encode'][0]=0.0
        elif bistrip=='Rarely':
            dataset['Business Travel encode'][0]=1.0
        else:
            dataset['Business Travel encode'][0]=2.0
        
        x = data[['TotalWorkingYears','Age','MonthlyIncome','JobLevel','StockOptionLevel','NumCompaniesWorked','Business Travel encode']]
        y = data['attrition encode']  
        x_train, x_test, y_train, y_test= train_test_split(x,y,test_size=0.3,random_state=45) 
        
        st.write('If you do not know what model do you want to choose, I give a link down below that explain each model')
        st.write('Learn more about logistic regression  : https://towardsdatascience.com/introduction-to-logistic-regression-66248243c148 ')
        st.write('Learn more about Naive Bayes Theorem  : https://towardsdatascience.com/naive-bayes-in-machine-learning-f49cc8f831b4 ')
        st.write('Learn more about Decision Tree        : https://towardsdatascience.com/decision-trees-in-machine-learning-641b9c4e8052 ')
        st.write('Learn more about Random Forest        : https://towardsdatascience.com/understanding-random-forest-58381e0602d2 ')
        
        st.write('But I suggest you to use logistic regression to predict since it has the highest accuracy compare to decision tree, SVM, naive bayes, even random forest')
        predict = st.selectbox("But in case you want to check, I made it for you. What model do you want to use?", ['Logistic Regression','Naive Bayes','Decision Tree','Random Forest'])
        if predict=='Logistic Regression':
            log_modelling(name)
        elif predict=='Naive Bayes':
            naivebayes_modelling(name)
        elif predict=='Decision Tree':
            decisiontree_modelling(name)
        elif predict=='Random Forest':
            randomforest_modelling(name)
           

### This is the start of function for data analytics
def analytics(feature):
    feature_head=data_numeric_short.sort_values(by = feature ).head(147)
    feature_tail=data_numeric_short.sort_values(by = feature ).tail(147)
    top_10= (feature_head[feature_head['Attrition'] == 'Yes'].count()[0]/147)*100
    bottom_10= (feature_tail[feature_tail['Attrition'] == 'Yes'].count()[0]/147)*100
    st.write("So there are " , top_10 , " % of resign employee in top 10% rank and " , bottom_10 , " % of resign employee in bottom 10% rank ")
    if (top_10 < 20.0) & (bottom_10 < 20.0) :
        st.write("It means " , feature ," doesn't really affect people to resign")
    elif (top_10 >= 20.0) :
        st.write("It means " , feature ," is one of the factor of attrition")
    elif (top_10 < 20.0) & (bottom_10 >=20.0) :
        st.write("It means " , feature ," is one of the factor of attrition but the correlation is negative")
    
    if st.button('Conclusion?'):
        st.write('I use only 7 features, they are : TotalWorkingYears,Age,MonthlyIncome,JobLevel,StockOptionLevel,NumCompaniesWorked and Business Travel, to predict the attrition. This decision is taken based on 3 reasons:')
        st.write('1. I am afraid to use model that I cannot explain. So instead just use feature importance in decision tree/random forest,I would like to do feature analysis by myself. Check out data analytics part to know how did I do that')
        st.write('2. I already validate the importance of these features using machine learning, and these features have a pretty good importance score too!')
        st.write('3. These choosen features I used are the things that HC can fill by themselves. Features like satisfaction index has to be filled by the employee')
        st.write('If you already have data like satisfaction index, I could add the feature here. Just contact me!')
### This is the end of function for data analytics
        
### This is the start of function for categorical data visualization    
def viz1(subject):       
    if subject=='Department':
        subject_x = 'Department_x'
        subject_y = 'Department_y'
    elif subject=='EducationField':
        subject_x = 'EducationField_x'
        subject_y = 'EducationField_y'
    elif subject=='Gender':
        subject_x = 'Gender_x'
        subject_y = 'Gender_y'
    elif subject=='JobRole':
        subject_x = 'JobRole_x'
        subject_y = 'JobRole_y'
    elif subject=='MaritalStatus':
        subject_x = 'MaritalStatus_x'
        subject_y = 'MaritalStatus_y'
        
    resign = pd.DataFrame(data.groupby([subject]).sum()['attrition encode'].reset_index())
    sub_all = pd.DataFrame(data[subject].value_counts().reset_index())

    sub_data = pd.merge(resign,sub_all,right_on='index', left_on= subject)
    sub_data['Not Resign'] = sub_data[subject_y] - sub_data['attrition encode']
    sub_data['Resign Percentage'] =  (sub_data['attrition encode']/sub_data[subject_y])*100
    sub_data['Not resign percentage'] = (sub_data['Not Resign']/sub_data[subject_y])*100

    
    plt.figure(figsize=[15,10])
    plt.bar(x=sub_data['index'], height=sub_data['attrition encode'])
    plt.bar(x=sub_data['index'], height=(sub_data['Not Resign']),bottom=sub_data['attrition encode'])
    
    plt.title("{} vs Number of Employee who Resign or not".format(subject))
    plt.legend(['Resign','Not Resign'])
    plt.xlabel(subject)
    plt.ylabel("Number of People")
    st.pyplot() 
    pd.options.display.float_format = '{:,.2f}'.format
    st.write(sub_data[[subject_x,subject_y,'Resign Percentage','Not resign percentage']].sort_values('Resign Percentage'))
###This is the end of function for categorical data visualization

###This is the definition of variable that we're using    
x = data[['TotalWorkingYears','Age','MonthlyIncome','JobLevel','StockOptionLevel','NumCompaniesWorked','Business Travel encode']]
y = data['attrition encode']  
x_train, x_test, y_train, y_test= train_test_split(x,y,test_size=0.3,random_state=45) 
###This is the end of variable definition 

###This is the start of function for logistic regression
def log_modelling(name):
    model_lr = LogisticRegression()
    model_lr.fit(x_train, y_train)
    accuracy = model_lr.score(x_test,y_test)
    st.write("Your employee with name " , name , " has ",  int(model_lr.predict_proba(dataset)[0][1] *100), "% chances to resign")
    st.write('This logistic regression model has accuracy :',accuracy*100,'%') 
###This is the end of function for logistic regression
    
###This is the start of function for naive bayes theorem
def naivebayes_modelling(name):
    model_nb = GaussianNB()
    model_nb.fit(x_train, y_train)
    accuracy1 = model_nb.score(x_test,y_test)
    st.write("Your employee with name " , name , " has ",  int(model_nb.predict_proba(dataset)[0][1] *100), "% chances to resign")
    st.write('This naive bayes model has accuracy :',accuracy1*100,'%') 
###This is the end of function for naive bayes theorem
    
###This is the start of function for decision tree 
def decisiontree_modelling(name):
    model_dt = DecisionTreeClassifier()
    model_dt.fit(x_train, y_train)
    accuracy2 = model_dt.score(x_test,y_test)
    if model_dt.predict(dataset)[0] == 0:
        status = ('will not resign')
    else:
        status = st.write('will resign')
    st.write("Your employee with name " , name , status)
    st.write('This decision tree model has accuracy :',accuracy2*100,'%') 
    st.write("Oh yeah and just in case you want to check whether feature I used is also important based on feature importance in decision tree, here you go :")
    featimp = pd.DataFrame(list(model_dt.feature_importances_), columns = ['Importances'])
    featcol = pd.DataFrame(list(x_train.columns), columns = ['Parameter'])
    featimp = featimp.join(featcol)
    featimp = pd.DataFrame(featimp.sort_values(by = ['Importances'], ascending = False))
    st.write("Feature importances : \n" , featimp)
###This is the end of function for decision tree

###This is the start of function for random forest
def randomforest_modelling(name):
    model_rf = RandomForestClassifier()
    model_rf.fit(x_train, y_train)
    accuracy3 = model_rf.score(x_test,y_test)
    if model_rf.predict(dataset)[0] == 0:
        status = ('will not resign')
    else:
        status = st.write('will resign')
    st.write("Your employee with name " , name , status)
    st.write('This random forest model has accuracy :',accuracy3*100,'%') 
    st.write("Oh yeah and just in case you want to check whether feature I used is also important based on feature importance in random forest, here you go :")
    
    feat_importances = pd.Series(model_rf.feature_importances_, index=x.columns)
    feat_importances = feat_importances.nlargest(20)
    feat_importances.plot(kind='barh')
    st.pyplot()
###This is the end of function for random forest
    
main()