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
    st.write(dataframe.columns)
    response = st.selectbox('Please select the response or dependent variable:',dataframe.columns)
    outlier_columns=st.multiselect('Please select the columns that you want to remove outliers',dataframe.columns)
    cat_columns =st.multiselect('Please select the categorical columns',dataframe.columns)
    automl = AutoPred(dataframe,response,outlier_columns,cat_columns)
    automl.remove_outlier_for_all_columns()
    st.write(automl.clean_df)
