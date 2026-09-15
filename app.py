import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score 
import streamlit as st
# this streamlit is for web based application project



# web page code
st.title("HEALTH INSURANCE PREDICTION")
img_url = "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTZatsR8RSbEFGssC4wrVuxi7PdcieCgJlov69Yc4MuPw&s=10"
st.image(img_url)


#LOAD DATA AND ML MODEL
# step 2: load data
url = "https://raw.githubusercontent.com/ankitmisk/UIT-data/refs/heads/main/Insurance.csv"
df = pd.read_csv(url)

# step 3: EDA: Exploratory Data Analysis
df.drop("Customer_ID", axis = 1, inplace = True)

df['Previous_Insurance'] = df['Previous_Insurance'].map({'No':0, 'Yes':1})
df['Insurance_Bought'] = df['Insurance_Bought'].map({'No':0, 'Yes':1})

#step 4 : divide dataset into features and target
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

#step 5 : divide data into training and testing part
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)

# step 6 : train model 
model = LogisticRegression()
model.fit(X_train, y_train)





#SHOW DATA SAMPLE
st.write(df.head())
#create side bar for user input form
st.sidebar.title("Fill Customer Details")
st.sidebar.image(img_url)

all_ans = []
for index, col_name in enumerate(X.columns):
  min_v = X[col_name].min()
  max_v = X[col_name].max()
  if col_name != "Previous_Insurance":
    value = st.sidebar.slider(f"Select value for {col_name}",
                              min_value = min_v,
                              max_value = max_v)
  else:
   value = st.sidebar.number_input(f"Select value for {col_name}:")

  all_ans.append(value)
  
ud = {j:all_ans[i] for i,j in enumerate(X.columns)}
user_df = pd.DataFrame(ud, index = [1])
st.write(user_df)



if st.button("Click to Predict: "):
    with st.spinner("Predicting.."):
        import time
        time.sleep(2)
    final_ans = model.predict([all_ans])[0]
    if final_ans == 0:
        st.write("Customer will not buy insurance")
    else:
        st.write("Customer will buy insurance")
    


  



