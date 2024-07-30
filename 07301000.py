from mlSandbox import MlSandbox

sb = MlSandbox("data/07301000.csv", 
               numerical=["establishmentYear","longitude","latitude","prev_percent_asset_capital","percent_asset_capital"], 
               numerical_log=["adj_assetsPrevMillion","adj_assetsMillion","adj_liabilitiesPrevMillion","adj_liabilitiesMillion","adj_incomePrevMillion","adj_incomeMillion","adj_expensesPrevMillion","adj_expensesMillion"], 
               one_hot=[],
               boolean=[], 
               objective="isPenalty", 
               objective_type="classification")

sb.linearRegression(random_state=2930)

# sb.neuralNetwork(epochs=150)