class AutoPred:

  def __init__(self,df,response,outlier_columns,cat_columns):
    self.raw_df = df
    self.response = response
    self.outlier_columns = outlier_columns
    self.cat_columns = cat_columns
  def remove_outlier(self, outlier_column):
    import numpy as np

    Q1 = np.percentile(self.raw_df[outlier_column], 25,
                    interpolation = 'midpoint')

    Q3 = np.percentile(self.raw_df[outlier_column], 75,
                    interpolation = 'midpoint')
    IQR = Q3 - Q1
    upper = np.where(self.raw_df[outlier_column] >= (Q3+1.5*IQR))
    # Lower bound
    lower = np.where(self.raw_df[outlier_column] <= (Q1-1.5*IQR))
    df_no_outlier = self.raw_df.drop(upper[0])
    df_no_outlier = df_no_outlier.drop(lower[0])
    df_no_outlier = df_no_outlier.reset_index(drop=True)
    return df_no_outlier
  def remove_outlier_for_all_columns(self):
    import streamlit as st
    for column in self.outlier_columns:
      self.clean_df = self.remove_outlier(column)
    st.write('All outliers are removed!')
    return self.clean_df
  def train_test_data(self,train_size):
    import h2o
    h2o.init()
    self.hf = h2o.H2OFrame(self.clean_df)
    for cat_col in self.cat_columns:
      self.hf[cat_col] = self.hf[cat_col].asfactor()
    if train_size >=1:
      df_train = self.clean_df.loc[:train_size,:]
      df_test = self.clean_df.loc[train_size:,:]
    if train_size>0 and train_size<1:
      df_train = self.clean_df.iloc[:int(len(self.clean_df)*train_size),:]
      df_test = self.clean_df.iloc[int(len(self.clean_df)*train_size):,:]
      print('The train size is {}.'.format(int(len(self.clean_df)*train_size)))
    else:
      raise TypeError('Invalid train size! The train size should  be integer or a proportion!')

    self.hf_train = h2o.H2OFrame(df_train)
    self.hf_test = h2o.H2OFrame(df_test)
    print('{} train and {} test data have been splited!'.format(len(self.hf_train),len(self.hf_test)))
    return self.hf,self.hf_train,self.hf_test
  def train(self):
    '''
    y: response variable
    hf_train: dataframe for training
    hf_test: dataframe for testing
    save_path: the path for saving the leader model

    '''
    from h2o.automl import H2OAutoML
    import streamlit as st
    st.write('test')
    self.aml = H2OAutoML(max_runtime_secs = 10)
    self.aml.train(y = self.response,
        training_frame = self.hf_train,
        leaderboard_frame=self.hf_test)
    self.leader_model = self.aml.leader
    st.write(self.aml.leaderboard)
    return self.leader_model
