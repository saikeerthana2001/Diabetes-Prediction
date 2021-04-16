import streamlit as st
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

model=open('gaussian_nb.pkl','rb')
model=pickle.load(model)

random=open('rff.pkl','rb')
random=pickle.load(random)


def pred(Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age):
    prediction=model.predict([[Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age]])
    return prediction
def predknn(Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age):
    dataset=pd.read_csv('diabetes.csv')
    val=[Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age]
    x=dataset.iloc[:,0:8]
    y=dataset.iloc[:,8]
    x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=0,test_size=0.1)
    a=MinMaxScaler()
    x_train=a.fit_transform(x_train)
    x_test=a.transform([val])
    classifier = KNeighborsClassifier(n_neighbors=13,p=2,metric='euclidean')
    classifier.fit(x_train,y_train)
    prediction=classifier.predict(x_test)
    return prediction
def predlr(Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age):
    dataset=pd.read_csv('diabetes.csv')
    val=[Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age]
    x=dataset.iloc[:,0:8]
    y=dataset.iloc[:,8]
    x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=0,test_size=0.1)
    a=MinMaxScaler()
    x_train=a.fit_transform(x_train)
    x_test=a.transform([val])
    regr= LogisticRegression(solver='lbfgs', max_iter=1000)
    regr.fit(x_train,y_train)
    prediction=regr.predict(x_test)
    return prediction
def predrf(Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age):
    prediction=random.predict([[Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age]])
    return prediction
def predsvm(Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age):
    dataset=pd.read_csv('diabetes.csv')
    val=[Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age]
    x=dataset.iloc[:,0:8]
    y=dataset.iloc[:,8]
    x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=0,test_size=0.1)
    a=MinMaxScaler()
    x_train=a.fit_transform(x_train)
    x_test=a.transform([val])
    support=SVC(kernel='linear')
    support.fit(x_train,y_train)
    prediction=support.predict(x_test)
    return prediction
    
st.title('Diabetes Prediction')
st.subheader('Please check your details')
preg=st.sidebar.slider("Pregnancies",0,10)
gluc=st.sidebar.slider("Glucose",1,200)
bp=st.sidebar.slider("Blood Pressure",1,100)
sthk=st.sidebar.slider("Skin Thickness",1,90)
ins=st.sidebar.slider("Insulin",1,200)
bmi=st.sidebar.slider("BMI",1,67)
dpf=st.sidebar.slider("Diabetes Pedigree Function",0.00,2.40)
age=st.sidebar.slider("Age",1,100)
st.write('Pregnancies: ',preg)
st.write('Glucose: ',gluc)
st.write('Blood Pressure: ',bp)
st.write('Skin Thickness: ',sthk)
st.write('Insulin: ',ins)
st.write('BMI: ',bmi)
st.write('Diabetes Pedigree Function: ',dpf)
st.write('Age: ',age)
a=st.button('Check using Naive Bayes')
b=st.button('Check using KNN')
c=st.button('Check using Logistic Regression')
d=st.button('Check using Random Forest')
e=st.button('Check using Support Vector Machine')
if a:
    result=pred(preg,gluc,bp,sthk,ins,bmi,dpf,age)
    if(result==[1]):
        st.error('You are diabetic')
    else:
        st.success('You are healthy')
    st.warning('Diabetes check done using Naive bayes algorithm')
if b:
    result=predknn(preg,gluc,bp,sthk,ins,bmi,dpf,age)
    if(result==[1]):
        st.error('You are diabetic')
    else:
        st.success('You are healthy')
    st.warning('Diabetes check done using KNN algorithm')
if c:
    result=predlr(preg,gluc,bp,sthk,ins,bmi,dpf,age)
    if(result==[1]):
        st.error('You are diabetic')
    else:
        st.success('You are healthy')
    st.warning('Diabetes check done using Logistic Regression algorithm')
if d:
    result=predrf(preg,gluc,bp,sthk,ins,bmi,dpf,age)
    if(result==[1]):
        st.error('You are diabetic')
    else:
        st.success('You are healthy')
    st.warning('Diabetes check done using Random Forest algorithm')
if e:
    result=predsvm(preg,gluc,bp,sthk,ins,bmi,dpf,age)
    if(result==[1]):
        st.error('You are diabetic')
    else:
        st.success('You are healthy')
    st.warning('Diabetes check done using Support Vector Machine algorithm')
    
    
    
    
