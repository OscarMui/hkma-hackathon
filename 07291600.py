from mlSandbox import MlSandbox

sb = MlSandbox("data/07291600.csv", 
               numerical=["establishmentYear","longitude","latitude"], 
               numerical_log=["adj_assetsPrevMillion","adj_assetsMillion","adj_liabilitiesPrevMillion","adj_liabilitiesMillion","adj_incomePrevMillion","adj_incomeMillion","adj_expensesPrevMillion","adj_expensesMillion"], 
               one_hot=[],
               boolean=[], 
               objective="isPenalty", 
               objective_type="classification")

sb.linearRegression()

# sb.neuralNetwork(epochs=150)