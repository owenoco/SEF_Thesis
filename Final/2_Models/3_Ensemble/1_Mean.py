#%%
# =========================
#Ensemble Mean Approach
# =========================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics

# =========================
#Import individual model predictions and forecasting target
# =========================
learner_predictions = pd.read_csv(r"C:\Users\Owen\OneDrive - Imperial College London\Research Project\Thesis\Final\2_Models\2_Individual\Learner_Predictions\12_month_learner_predictions.csv", index_col=[0])
df = pd.read_csv(r"C:\Users\Owen\OneDrive - Imperial College London\Research Project\Thesis\Data Collection\feature_engineered_data_2.csv", index_col=[0])
learner_predictions = np.asarray(learner_predictions)
df = np.asarray(df)
y_actual = df[2880:20160, -1]

# =========================
#Calculate Mean Forecast at each SP
# =========================
mean_preds = np.mean(learner_predictions, axis=1)

# =========================
#Mean Ensemble Evaluation
# =========================
Mean_RMSE = metrics.mean_squared_error(y_actual, mean_preds, squared = False)
Mean_MAE = metrics.mean_absolute_error(y_actual, mean_preds)
residual = np.abs(y_actual-mean_preds)
print("Mean Ensemble RMSE: " + str(Mean_RMSE))
print("Mean Ensemble MAE: " + str(Mean_MAE))

#Plot
plt.figure(figsize=(12, 6))
sns.set_style("darkgrid")
plt.title('Mean Ensemble - First Five Days').set_fontsize(20)
plt.plot(y_actual[:240],'--', label="Actual Value", color='#8c8488')
plt.plot(mean_preds[:240], label="Mean Ensemble", color='#0a4236')
plt.plot(residual[:240], label="Residual", color='red', alpha=0.2)
plt.legend(loc=1)
plt.ylabel('System Sell Price (£/MWh)').set_fontsize(16)
plt.xlabel('Settlement Period').set_fontsize(16)
plt.show(block=False)

# =========================
#Export Test Predections and Residuals
# =========================
test_preds = pd.read_csv(r"C:\Users\Owen\OneDrive - Imperial College London\Research Project\Thesis\Final\2_Models\3_Ensemble\ensemble_predictions.csv", index_col=[0])
test_preds.iloc[:,0] = mean_preds
test_preds.to_csv(r"C:\Users\Owen\OneDrive - Imperial College London\Research Project\Thesis\Final\2_Models\3_Ensemble\ensemble_predictions.csv")

test_preds = pd.read_csv(r"C:\Users\Owen\OneDrive - Imperial College London\Research Project\Thesis\Final\2_Models\3_Ensemble\ensemble_errors.csv", index_col=[0])
test_preds.iloc[:,0] = residual
test_preds.to_csv(r"C:\Users\Owen\OneDrive - Imperial College London\Research Project\Thesis\Final\2_Models\3_Ensemble\ensemble_errors.csv")
# %%
