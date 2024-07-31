import numpy as np
from mlSandbox import MlSandbox
import matplotlib.pyplot as plt

numerical = ["establishmentYear","longitude","latitude","prev_percent_asset_capital","percent_asset_capital"]
numerical_log = ["adj_assetsPrevMillion","adj_assetsMillion","adj_liabilitiesPrevMillion","adj_liabilitiesMillion","adj_incomePrevMillion","adj_incomeMillion","adj_expensesPrevMillion","adj_expensesMillion"]
sb = MlSandbox("data/07310929.csv", 
               numerical=numerical, 
               numerical_log=numerical_log, 
               one_hot=[],
               boolean=[], 
               objective="isPenalty", 
               objective_type="classification")

base = 4096

#! linear regression
weights = []
mses = []
base = 4096
for i in range(15):
    print(f"---- {i} ----")
    (w,e) = sb.linearRegression(random_state=i+base)
    weights.append(w)
    mses.append(e)


average_weights = np.mean(weights, axis=0)

print("---- final ----")
for feature, weight in zip(numerical+numerical_log, average_weights):
            print(f"{feature}: {weight}")

print(f"Average error: {sum(mses)/len(mses)}")

#! NN
# for i in range(5):
#     sb.neuralNetwork(epochs=150)