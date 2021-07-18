import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import streamlit as st


st.title("Iris Classification Using Decision Tree")
nav = st.sidebar.radio("Navigation", ["Home", "About Data", " Train & Prediction"])
df=pd.read_csv("Iris.csv")

# One Hot Encoding
from sklearn.preprocessing import OneHotEncoder
oh_enc=OneHotEncoder(sparse=False)
oh_enc_arr=oh_enc.fit_transform(df[["Species"]])
on_enc_df=pd.DataFrame(oh_enc_arr,columns=["Iris-setosa","Iris-versicolor","Iris-virginica"])   # Converting array in Dataframe
final_df=pd.merge(df,on_enc_df,left_index=True,right_index=True)                                #merging both dataframe
final_df.drop(["Species","Id"],axis=1,inplace=True)                                             #removing unnecessary columns

# Split Data
X=final_df.iloc[:,0:4]
Y=final_df.iloc[: ,4:7]
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.20,random_state=0)

def testtraindimensions():
    st.write("Shape of X_train",X_train.shape)
    st.write("Shape of X_test",X_test.shape)
    st.write("Shape of Y_train",Y_train.shape)
    st.write("Shape of Y_test",Y_test.shape)



if nav=="Home":
    st.header("Iris For Classification ")
    col1, col2, col3 = st.beta_columns(3)
    iris_setosa = Image.open("Iris_setosa.jpg")
    col1.header("Iris_setosa")
    col1.image(iris_setosa, use_column_width=True, width=100)
    iris_versicolor= Image.open("Iris_versicolor.jpg")
    col2.header("Iris_versicolor")
    col2.image(iris_versicolor, use_column_width=True, width=100)
    iris_virginica= Image.open("Iris_virginica.jpg")
    col3.header("Iris_virginica")
    col3.image(iris_virginica, use_column_width=True, width=100)
    st.header("Objective")
    st.write("Create the Decision Tree classifier and visualize it graphically.We will give new  data to this classifier, it would be able to predict the right class accordingly.")
if nav=="About Data":
    data= st.selectbox("Show Data", ("Head", "Tail"))
    if data=="Head":
        st.write("Top Five rows")
        st.table(df.head())
    if data=="Tail":
        st.write("Last Five rows")
        st.table(df.tail())
    if st.checkbox('Null Values'):
        st.write("Null Values")
        st.write(df.isnull().sum())
    if st.checkbox("Data Description"):
        st.write(df.describe())
    if st.checkbox("Coorelation"):
        st.write(df.corr())
        fig, ax = plt.subplots()
        sns.heatmap(df.corr(), cmap="YlGnBu", ax=ax)
        st.write(fig)

if nav==" Train & Prediction":
     from sklearn.tree import DecisionTreeClassifier
     classifier=DecisionTreeClassifier(criterion="gini")
     if st.checkbox("Train Model"):
        testtraindimensions()
        classifier.fit(X_train,  Y_train)
        st.write("Model Successfully Trained ")
     if st.checkbox("Predict"):
        SepalLength=st.slider("SepalLengthCm", 4.0, 8.0, 0.1)
        SepalWidth=st.slider("SepalWidthCm", 2.0, 4.5, 0.1)
        PetalLength=st.slider("PetalLengthCm", 1.0, 7.0, 0.1)
        PetalWidth=st.slider("SepalWidthCm", 0.1, 3.0, 0.1)
        if st.button("Classify"):
            result=classifier.predict([[SepalLength, SepalWidth, PetalLength, SepalWidth]])
            if result[0][0] ==1:
                st.write("Iris_Setosa")
                iris_setosa = Image.open("Iris_setosa.jpg")
                st.image(iris_setosa, width=400)
            if result[0][1]==1:
                st.write("Iris_Versicolor")
                iris_versicolor = Image.open("Iris_versicolor.jpg")
                st.image(iris_versicolor, width=400)
            if result[0][2]==1:
                st.write("Iris_Virginica")
                iris_virginica = Image.open("Iris_virginica.jpg")
                st.image(iris_virginica, width=400)


