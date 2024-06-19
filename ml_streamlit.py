 #) importing necessary module
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
import streamlit as st

#) dataframe
df = pd.DataFrame()
df['x1_value'] = [1,5.5,2,4.5,3,2.5,4,1.5,5,3.5,11,15,12.5,14.5,13,12,14,11.5,15.5,13.5]
df['x2_value'] = [100,125,105,110,115,120,123,113,107,119,490,497,485,489,493,484,480,487,475,479]
df['group'] = [1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2,]
    
#)logistic regression
X = df.drop(['group'],axis=1)
y = df['group']
x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.2)
model = LogisticRegression()
model.fit(x_train,y_train)
train_pred = model.predict(x_train)
test_pred = model.predict(x_test)
    
st.title(":orange[logistic regression prediction]")

with st.form("logistic regression predictor"):
    col1,col2,col3 = st.columns([3,2,1])
with col1:
    x1 = st.number_input("x1 value")
with col2:
    x2 = st.number_input("x2 value")
with col3:
    submit = st.form_submit_button('Predict')

if submit:
    new_input = [[x1, x2]]
    
    # get prediction for new input
    new_output = model.predict(new_input)

    # classifying the new data
    st.subheader(":green[the class of new data]\n")
    st.write(new_output)