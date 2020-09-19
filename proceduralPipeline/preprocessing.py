import pandas as pd 
import numpy as np 
import joblib
import os
import logging
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error,r2_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import Lasso

class Pipeline():

    def __init__(self, filepath,scaler_output="scaler.pkl",model_output="model.pkl",target_variable="",split_percentage=0.1,rare_percentage=0.01, missing_categorical = [], missing_numerical = [], temp_variables = [], log_variables = [],selected_features=[],
        exclude_variables=[],from_column="",replace_categorical="", replace_numerical="", encoding_infrequent="",seed=0,**kwargs):
        
        self.data = self.load_data(filepath, **kwargs)
        self.train_X = None
        self.train_Y = None 
        self.test_X = None
        self.test_Y = None
        self.mse = None
        self.r2 = None
        self.rmse = None
        self.mae = None
        self.cat_vars = []
        self.cat_mapping = {}
        self.target_variable = target_variable
        self.seed = seed
        self.split_percentage = split_percentage
        self.rare_percentage = rare_percentage
        self.missing_categorical = missing_categorical
        self.missing_numerical = missing_numerical
        self.replace_categorical = replace_categorical
        self.replace_numerical = replace_numerical
        self.from_column = from_column
        self.temp_variables = temp_variables
        self.encoding_infrequent = encoding_infrequent
        self.log_variables = log_variables
        self.exclude_variables = exclude_variables
        self.selected_features = selected_features
        self.scaler_output = scaler_output
        self.model_output = model_output
        self.model = Lasso(alpha=0.005,random_state=self.seed)
        self.predictions = []
        
    
    def load_data(self,filepath,**kwargs):
        df = pd.read_csv(filepath, **kwargs)
        return df

    def divide_data_train_test(self, data,target, seed, split_percentage=0.1):
        train_X, test_X, train_Y, test_Y = train_test_split(data, data[target],test_size=split_percentage, random_state=seed)
        return train_X, train_Y, test_X, test_Y

    def fill_missing_cat(self, data, columns, replace="Missing"):
        df = data.copy()
        df[columns] = df[columns].fillna(replace)
        return df

    def fill_missing_num(self, traindata, testdata, columns, replace="mode"):
        df_train = traindata.copy()
        df_test = testdata.copy()
        for col in columns:
            if replace == "mode":
                replace_value = df_train[col].aggregate(replace)[0]
            else:
                replace_value = df_train[col].aggregate(replace)
            
            #trainData
            df_train[col + "_na"] = np.where(df_train[col].isnull(),1,0)
            df_train[col].fillna(replace_value, inplace=True)

            #testData
            df_test[col + "_na"] = np.where(df_test[col].isnull(),1,0)
            df_test[col].fillna(replace_value, inplace=True)
        
        return df_train, df_test

    def elapsed_years(self, data, temporal_columns, from_column):
        df = data.copy()
        for col in temporal_columns:
            if col not in from_column:
                df[col] = df[from_column] - df[col]
        return df

    def find_frequent_labels(self, data, var, rare_perc=0.01):
        df = data.copy()
        tmp = df[var].value_counts()/df.shape[0]
        return tmp[tmp>rare_perc].index
    
    def encode_infrequent_labels(self, train_X,test_X, rare_percentage=0.01, coding="Rare"):
        df_train = train_X.copy()
        df_test = test_X.copy()
        self.cat_vars = [var for var in df_train if df_train[var].dtype == "O"]
        for var in self.cat_vars:
            frequent_ls = self.find_frequent_labels(df_train, var, rare_percentage)
            df_train[var] = np.where(df_train[var].isin(frequent_ls),df_train[var],coding)
            df_test[var] = np.where(df_test[var].isin(frequent_ls),df_test[var],coding)
        return df_train, df_test

    def categorical_encoding(self, train, cat_variables, target_variable):
        df_train = train.copy()
        
        for var in cat_variables:
            ordered_labels = df_train.groupby(var)[target_variable].mean().sort_values().index
            mapping = {k:i for i,k in enumerate(ordered_labels,0)}

            self.cat_mapping[var] = mapping 
        
        return self
        

    def log_transformation(self, data, columns):
        df = data.copy()
        for col in columns:
            df[col] = np.log(df[col])

        return df
    
    def train_scaler(self, trainData, outputpath):
        scaler = MinMaxScaler()
        scaler.fit(trainData)
        joblib.dump(scaler, outputpath)
        return None

    def scale_features(self, data, scaler):
        df = data.copy()
        scaler = joblib.load(scaler)
        return scaler.transform(df)

    def train_model(self, trainData, trainTarget, outputpath, seed):

        self.model.fit(trainData, trainTarget)

        print("Model training finished!")

        joblib.dump(self.model, outputpath)
        return None

    def predict_model(self, testData, model):

        model = joblib.load(model)
        return model.predict(testData)


    def process_data(self):
        self.train_X, self.train_Y, self.test_X, self.test_Y = self.divide_data_train_test(self.data, self.target_variable,self.seed, self.split_percentage)
        self.train_X, self.test_X = self.fill_missing_cat(self.train_X,self.missing_categorical,self.replace_categorical), self.fill_missing_cat(self.test_X,self.missing_categorical,self.replace_categorical)
        self.train_X, self.test_X = self.fill_missing_num(self.train_X, self.test_X, self.missing_numerical, self.replace_numerical)
        self.train_X, self.test_X = self.elapsed_years(self.train_X, self.temp_variables,self.from_column), self.elapsed_years(self.test_X, self.temp_variables,self.from_column)
        return self

    def transform_data(self):
        self.train_X, self.test_X = self.encode_infrequent_labels(self.train_X, self.test_X, self.rare_percentage, self.encoding_infrequent) 
        self.train_X, self.test_X = self.log_transformation(self.train_X, self.log_variables),self.log_transformation(self.test_X,self.log_variables)
        self.categorical_encoding(self.train_X, self.cat_vars, self.target_variable)
        
        for var in self.cat_vars:
            self.train_X[var] = self.train_X[var].map(self.cat_mapping[var])
            self.test_X[var] = self.test_X[var].map(self.cat_mapping[var])
        
        self.train_vars = [var for var in self.train_X if var not in self.exclude_variables]
        self.train_scaler(self.train_X[self.train_vars],self.scaler_output)
        self.train_X[self.train_vars], self.test_X[self.train_vars] = self.scale_features(self.train_X[self.train_vars], self.scaler_output), self.scale_features(self.test_X[self.train_vars], self.scaler_output)
        return self

    
    def export_train_test(self):
        self.train_X.to_csv("xtrain.csv", index=False)
        self.test_X.to_csv("xtest.csv",index=False)

    def train(self):
        self.train_model(self.train_X[self.selected_features], self.train_X[self.target_variable], self.model_output, seed=self.seed)

    def predict(self):
        self.predictions = self.predict_model(self.test_X[self.selected_features],self.model_output)
        return self
    
    def score(self):
        self.mse = mean_squared_error(np.exp(self.test_X[self.target_variable]),np.exp(self.predictions))
        self.rmse = np.sqrt(self.mse)
        self.r2 = r2_score(np.exp(self.test_X[self.target_variable]),np.exp(self.predictions))
        self.mae = mean_absolute_error(np.exp(self.test_X[self.target_variable]),np.exp(self.predictions))

        print("MSE is {}, RMSE is {}, MAE is {} and r2 is {}" .format(self.mse, self.rmse, self.mae, self.r2))

        return self


