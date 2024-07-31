import numpy as np
from mlSandbox import MlSandbox
import matplotlib.pyplot as plt
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

numerical = ["establishmentYear","longitude","latitude","prev_percent_asset_capital","percent_asset_capital","assetsPrecentChange","liabilitiesPrecentChange","incomePercentChange","expensesPercentChange","profitPercentChange"]
numerical_log = ["adj_assetsPrevMillion","adj_assetsMillion","adj_liabilitiesPrevMillion","adj_liabilitiesMillion","adj_incomePrevMillion","adj_incomeMillion","adj_expensesPrevMillion","adj_expensesMillion"]
sb = MlSandbox("data/07311200.csv", 
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
biases = []
base = 4096
for i in range(15):
    # print(f"---- {i} ----")
    (w,b,e) = sb.linearRegression(random_state=i+base)
    weights.append(w)
    mses.append(e)
    biases.append(b)


average_weights = np.mean(weights, axis=0)

print("---- final results ----")
for feature, weight in zip(numerical+numerical_log, average_weights):
            print(f"{feature}: {weight}")
# print(f"bias: {sum(biases)/len(biases)}")
averageError = sum(mses)/len(mses)
averageAccuracy = 1 - averageError
print(f"Average Error: {averageError:.2f}")
print(f"Average Accuracy: {averageAccuracy:.2f}")

# Create a figure and axis
fig, ax = plt.subplots(figsize=(8, 6))

# Plot the weights against the features
ax.bar(range(len(average_weights)), average_weights)
ax.set_xticks(range(len(numerical+numerical_log)))
ax.set_xticklabels(["EST","LON","LAT","PC/A","C/A","ΔA","ΔL","ΔI","ΔE","ΔP","PA","A","PL","L","PI","I","PE","E"])
ax.set_xlabel('Feature')
ax.set_ylabel('Weight')
ax.set_title('Weights vs Features')

# Show the plot
# plt.show()
fig, ax = plt.subplots(figsize=(6, 4), dpi=80)  # Adjust figsize and dpi for smaller plot

# Plot the weights against the features
ax.bar(range(len(average_weights)), average_weights)
ax.set_xticks(range(len(numerical+numerical_log)))
ax.set_xticklabels(["EST","LON","LAT","PC/A","C/A","ΔA","ΔL","ΔI","ΔE","ΔP","PA","A","PL","L","PI","I","PE","E"])
ax.set_xlabel('Feature')
ax.set_ylabel('Weight')
ax.set_title('Weights vs Features')

# Save the plot with smaller size
plt.savefig('plot.png', bbox_inches='tight')
# plt.savefig('plot.png')

#! NN
# for i in range(5):
#     sb.neuralNetwork(epochs=150)

#! SVM
# accuracies = []

# for i in range(10):
#     accuracy = sb.svm(random_state=308+i)  # Changing the random state to vary the splits
#     accuracies.append(accuracy)

# average_accuracy = sum(accuracies) / len(accuracies)
# print(f"Average Accuracy over 10 runs: {average_accuracy:.2f}")