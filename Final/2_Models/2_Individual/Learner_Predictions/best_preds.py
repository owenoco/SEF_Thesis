#%%
# =========================
#Plotting the best individual predictions for each SP from IndividualModels
# =========================
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# =========================
#Import Individual predictions and their residuals
# =========================
learner_predictions = pd.read_csv(r"C:\Users\Owen\OneDrive - Imperial College London\Research Project\Thesis\Final\2_Models\2_Individual\Learner_Predictions\12_month_learner_predictions.csv", index_col=[0])
learner_predictions = np.asarray(learner_predictions)

learner_errors = pd.read_csv(r"C:\Users\Owen\OneDrive - Imperial College London\Research Project\Thesis\Final\2_Models\2_Individual\Learner_Errors\12_month_learner_errors.csv", index_col=[0])
learner_errors = np.asarray(learner_errors)

# =========================
#Determine best prediction for each SP
# =========================
best_idx = np.argmin(learner_errors, axis=1)
best_preds = np.choose(best_idx, learner_predictions.T)
best_encoded = pd.get_dummies(pd.Series(best_idx))

# =========================
#Plot Results as a Heatmap
# =========================
plt.figure(figsize=(12, 6))
sns.set_style("darkgrid")
plt.title('Best Predictions').set_fontsize(20)
plt.xlabel('Settlement Period').set_fontsize(16)
y_axis_labels = [ 'RFR', 'SVR', 'FFNN', 'GRU', 'LSTM', 'Naive'] # labels for y-axis
g = sns.heatmap(best_encoded.transpose(), yticklabels=y_axis_labels, cmap="PuBuGn", cbar=False)
plt.show(block=False)

# %%
