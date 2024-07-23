from mlSandbox import MlSandbox

sb = MlSandbox("data/housing.csv", 
               numerical=["CHAS","ZN","INDUS","NOX","RM","AGE","DIS","RAD","TAX","PTRATIO","B","LSTAT","UNK"], 
               numerical_log=[], 
               one_hot=[],
               boolean=[], 
               objective="CRIM", 
               objective_type="regression")

# sb.linearRegression()

sb.neuralNetwork(epochs=100)