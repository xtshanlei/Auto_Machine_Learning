import streamlit as st
from Auto_Pred import AutoPred
import pandas as pd
import numpy  as np
import seaborn as sns

#######Upload data files#######
st.header('1. Upload dataset')
uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    dataframe = pd.read_csv(uploaded_file)
    st.write(dataframe)
    st.header('2. Preprocessing')
    response = st.selectbox('Please select the response or dependent variable:',dataframe.columns)
    if response:
        cat_columns =st.multiselect('Please select the categorical columns',dataframe.columns)
        if cat_columns:
            outlier_columns=st.multiselect('Please select the columns that you want to remove outliers',dataframe.columns)
            if outlier_columns:
                automl = AutoPred(dataframe,response,outlier_columns,cat_columns)
                automl.remove_outlier_for_all_columns()
                st.write('The data without outliers are shown below:')
                st.write(automl.clean_df)
                st.header('3. Train test split')
                train_size = st.slider('The percentage of training set?',0.0,1.0,0.1)
                if st.button('Split the dataset!'):
                    automl.train_test_data(train_size)
                    st.write('The dataset has been split into {} training and {} test samples'.format(len(automl.hf_train),len(automl.hf_test)))
                    st.header('5. Training')
                    save_path = '/h2o.zip'
                    st.write(save_path)
                    leader_model = automl.train(save_path)
                    st.download_button('Click to download the trained model', leader_model)
