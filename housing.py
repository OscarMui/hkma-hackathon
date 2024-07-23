from mlSandbox import MlSandbox

sb = MlSandbox("data/housing.csv", 
               numerical=["CRIM","CHAS","ZN","INDUS","NOX","RM","AGE","DIS","RAD","TAX","PTRATIO","B","LSTAT"], 
               numerical_log=[], 
               one_hot=[],
               boolean=[], 
               objective="MEDV", 
               objective_type="regression")

# sb.linearRegression()

sb.neuralNetwork(epochs=300)