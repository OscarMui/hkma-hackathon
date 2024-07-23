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
        """
        Performs linear regression on the data.
        
        Parameters:
        - test_size (float, optional): The proportion of the dataset to include in the test split. Defaults to 0.2.
        - random_state (int, optional): The seed used by the random number generator. Defaults to 42.
        
        Returns:
        - model: the trained linear regression model
        - score: the R-squared score of the model
        """
        # Load the data
        df = pd.read_csv(self.file_name)
        
        # Preprocess the data
        X = df[self.numerical + self.numerical_log + self.one_hot + self.boolean]
        y = df[self.objective]
        
        # Split the data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        
        # Define the preprocessing steps
        categorical_transformer = OneHotEncoder()
        numerical_transformer = StandardScaler()
        numerical_log_transformer = Pipeline([
            ('log', FunctionTransformer(np.log1p, validate=True)),
            ('scaler', StandardScaler())
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ('categorical', categorical_transformer, self.one_hot),
                ('numerical', numerical_transformer, self.numerical),
                ('numerical_log', numerical_log_transformer, self.numerical_log)
            ]
        )

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

        # check mean squared error
        test_predictions = model.predict(X_test)
        print("y_test: ",y_test[:5])
        print("predictions: ",test_predictions[:5])
        mse = mean_squared_error(y_test, test_predictions)
        print(f"Mean Squared Error: {mse:.2f}")