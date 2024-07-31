from datetime import datetime
import joblib
import pickle
import numpy as np
import pandas as pd
from sklearn.calibration import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.discriminant_analysis import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FunctionTransformer, Pipeline
from sklearn.preprocessing import OneHotEncoder

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import save_model

from scikeras.wrappers import KerasRegressor

class MlSandbox:
    def __init__(self, file_name, numerical=[], numerical_log=[], one_hot=[], boolean=[], objective=None, objective_type="regression"):
        assert(objective != None and (objective_type == "regression" or objective_type == "classification"))
        self.file_name = file_name
        self.numerical = numerical
        self.numerical_log = numerical_log
        self.one_hot = one_hot
        self.boolean = boolean
        self.objective = objective
        self.objective_type = objective_type
        
    def linearRegression(self, test_size=0.1, random_state=42):
        # Load the data
        df = pd.read_csv(self.file_name)
        
        # Preprocess the data
        X = df[self.numerical + self.numerical_log + self.one_hot + self.boolean]
        y = df[self.objective]
        
        print(X)
        # Split the data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        
        # Define the preprocessing steps
        preprocessor = self.__createPreprocessor()

        # Create the pipeline
        model = Pipeline([
            ('preprocess', preprocessor),
            ('regressor', LinearRegression()) if self.objective_type=="regression" else ("regressor",LogisticRegression())
        ])
        
        # Fit the model
        model.fit(X_train, y_train)
        
        # Print the weights
        print("Regression Weights:")
        features = X.columns.values
        weights = model['regressor'].coef_[0]
        # print(weights)
        for feature, weight in zip(features, weights):
            print(f"{feature}: {weight}")
        print(f"bias: {model['regressor'].intercept_}")
        print("")

        # check mean squared error
        test_predictions = model.predict(X_test)
        print("y_test: ",y_test[:5])
        print("predictions: ",test_predictions[:5])
        mse = mean_squared_error(y_test, test_predictions)
        print(f"Mean Squared Error: {mse:.2f}")

        return (weights, mse)

    def neuralNetwork(self,epochs=100, batch_size=32, optimizer='adam',loss='mean_squared_error', test_size=0.2, random_state=42):
        # Load the data
        df = pd.read_csv(self.file_name)
        
        # Preprocess the data
        X = df[self.numerical + self.numerical_log + self.one_hot + self.boolean]
        y = df[self.objective]
        
        # Split the data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        
        # Define the preprocessing steps
        preprocessor = self.__createPreprocessor()

        def createModel():
            input_dim = X.shape[1]
            model_ = Sequential()
            model_.add(Dense(64, input_dim=input_dim, activation='relu'))
            model_.add(Dense(32, activation='relu'))
            if self.objective_type == "regression":
                model_.add(Dense(1))
            else:
                model_.add(Dense(1, activation='sigmoid'))
            model_.compile(optimizer=optimizer, loss=loss)
            return model_
        
        # Create the pipeline
        model = Pipeline([
            ('preprocess', preprocessor),
            ('regressor', KerasRegressor(model=createModel(), epochs=epochs, batch_size=batch_size))
        ])
        
        # Fit the model
        model.fit(X_train, y_train)

        # check mean squared error
        test_predictions = model.predict(X_test)
        print("y_test: ",y_test[:5])
        print("predictions: ",test_predictions[:5])
        mse = mean_squared_error(y_test, test_predictions)
        print(f"Mean Squared Error: {mse:.2f}")

        # Save the model weights
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        model.named_steps['regressor'].model.save(f'model_weights_{timestamp}.keras')
        print("Model weights saved to", f'model_weights_{timestamp}.keras')

    def __createPreprocessor(self):
        categorical_transformer = OneHotEncoder()
        numerical_transformer = StandardScaler()
        numerical_log_transformer = Pipeline([
            ('log', FunctionTransformer(np.log1p, validate=True)),
            ('scaler', StandardScaler())
        ])

        return ColumnTransformer(
            transformers=[
                ('categorical', categorical_transformer, self.one_hot),
                ('numerical', numerical_transformer, self.numerical),
                ('numerical_log', numerical_log_transformer, self.numerical_log)
            ]
        )
    
    def svm(self, test_size=0.25, random_state=42):
        # Load the data
        df = pd.read_csv(self.file_name)
        
        # Preprocess the data
        X = df[self.numerical + self.numerical_log + self.one_hot + self.boolean]
        y = df[self.objective]
        
        print(X)
        # Split the data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        
        # Define the preprocessing steps
        preprocessor = self.__createPreprocessor()

        # Create the pipeline
        model = Pipeline([
            ('preprocess', preprocessor),
            ('classifier', SVC())
        ])
        
        # Fit the model
        model.fit(X_train, y_train)
        
        # Print the support vectors
        # print("Number of support vectors for each class:")
        # print(model['classifier'].n_support_)
        # print("Support vectors:")
        # print(model['classifier'].support_vectors_)

        # Check accuracy
        test_predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, test_predictions)
        print(f"Accuracy: {accuracy:.2f}")
        return accuracy
