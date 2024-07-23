import datetime
import numpy as np
import pandas as pd
from sklearn.calibration import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.discriminant_analysis import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FunctionTransformer, Pipeline
from sklearn.preprocessing import OneHotEncoder

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

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
        
    def linearRegression(self, test_size=0.2, random_state=42):
        # Load the data
        df = pd.read_csv(self.file_name)
        
        # Preprocess the data
        X = df[self.numerical + self.numerical_log + self.one_hot + self.boolean]
        y = df[self.objective]
        
        # Split the data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        
        # Define the preprocessing steps
        preprocessor = self.__createPreprocessor()

        # Create the pipeline
        model = Pipeline([
            ('preprocess', preprocessor),
            ('regressor', LinearRegression())
        ])
        
        # Fit the model
        model.fit(X_train, y_train)
        
        # Print the weights
        print("Regression Weights:")
        features = model['preprocess'].get_feature_names_out()
        weights = model['regressor'].coef_
        for feature, weight in zip(features, weights):
            print(f"{feature}: {weight}")
        print(f"bias: {model['regressor'].intercept_}")
        print("")

        # Evaluate the model
        # score = model.score(X_test, y_test)
        # print("Test Score:", score)

        # check mean squared error
        test_predictions = model.predict(X_test)
        print("y_test: ",y_test[:5])
        print("predictions: ",test_predictions[:5])
        mse = mean_squared_error(y_test, test_predictions)
        print(f"Mean Squared Error: {mse:.2f}")

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
            model_.add(Dense(1))
            model_.compile(optimizer=optimizer, loss=loss)
            return model_
        
        # Create the pipeline
        model = Pipeline([
            ('preprocess', preprocessor),
            ('regressor', KerasRegressor(model=createModel, epochs=epochs, batch_size=batch_size))
        ])
        
        # Fit the model
        model.fit(X_train, y_train)

        # Evaluate the model
        # score = model.score(X_test, y_test)
        # print("Test Score:", score)

        # check mean squared error
        test_predictions = model.predict(X_test)
        print("y_test: ",y_test[:5])
        print("predictions: ",test_predictions[:5])
        mse = mean_squared_error(y_test, test_predictions)
        print(f"Mean Squared Error: {mse:.2f}")

        # model.named_steps['regressor'].save_weights()

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
    

# class KerasRegressor(BaseEstimator, TransformerMixin):
#     def __init__(self, epochs=100, batch_size=32, optimizer='adam', loss='mean_squared_error', verbose=0):
#         self.epochs = epochs
#         self.batch_size = batch_size
#         self.optimizer = optimizer
#         self.loss = loss
#         self.verbose = verbose
#         self.model_ = None

#     def fit(self, X, y):
#         input_dim = X.shape[1]
#         self.model_ = Sequential()
#         self.model_.add(Input(shape=(input_dim,)))
#         self.model_.add(Dense(64, input_dim=input_dim, activation='relu'))
#         self.model_.add(Dense(32, activation='relu'))
#         self.model_.add(Dense(1))
#         self.model_.compile(optimizer=self.optimizer, loss=self.loss)
#         self.model_.fit(X, y, epochs=self.epochs, batch_size=self.batch_size, verbose=self.verbose)
#         return self

#     def predict(self, X):
#         return self.model_.predict(X)

#     def score(self, X, y):
#         return self.model_.evaluate(X, y, verbose=0)
    
#     def save_weights(self):
#         if self.model_ is not None:
#             self.model_.save_weights(f"model_{self.file_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.h5")
#             print(f"Model weights saved to: model_{self.file_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.h5")
#         else:
#             print("No model to save. Please train the model first.")