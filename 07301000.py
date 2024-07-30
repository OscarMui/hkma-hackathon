import numpy as np
from mlSandbox import MlSandbox

numerical = ["establishmentYear","longitude","latitude","prev_percent_asset_capital","percent_asset_capital"]
numerical_log = ["adj_assetsPrevMillion","adj_assetsMillion","adj_liabilitiesPrevMillion","adj_liabilitiesMillion","adj_incomePrevMillion","adj_incomeMillion","adj_expensesPrevMillion","adj_expensesMillion"]
sb = MlSandbox("data/07301000.csv", 
               numerical=numerical, 
               numerical_log=numerical_log, 
               one_hot=[],
               boolean=[], 
               objective="isPenalty", 
               objective_type="classification")

weights = []
for i in range(10):
    print(f"---- {i} ----")
    weights.append(sb.linearRegression(random_state=i))


average_weights = np.mean(weights, axis=0)

print("---- final ----")
for feature, weight in zip(numerical+numerical_log, average_weights):
            print(f"{feature}: {weight}")

# sb.neuralNetwork(epochs=150)