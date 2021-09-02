#%%
# =========================
#ES Yesterday Ensemble Approach
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
learner_residual = pd.read_csv(r"C:\Users\Owen\OneDrive - Imperial College London\Research Project\Thesis\Final\2_Models\2_Individual\Learner_Errors\12_month_learner_errors.csv", index_col=[0])
df = pd.read_csv(r"C:\Users\Owen\OneDrive - Imperial College London\Research Project\Thesis\Data Collection\feature_engineered_data_2.csv", index_col=[0])

learner_predictions = np.asarray(learner_predictions)
learner_residual = np.asarray(learner_residual)
df = np.asarray(df)

# =========================
#Calculate and applying the weights to individual model forecasts
# =========================
preds = []
ensemble_weights = []

for i in range(len(learner_predictions)):
    #Weights assigned randomly for the first day
    if i <= 51:
        random_model = np.zeros(5, dtype=np.int8).tolist()
        random_model[np.random.randint(0,5)] = 1 
        preds.insert(i, np.sum(np.multiply(learner_predictions[i], random_model)))
        ensemble_weights.append(random_model)
    
    #Weights based off MSPE ffrom the previous day
    else: 
        MSPE = np.divide(np.sum(np.square(learner_residual[i-51:i-3]), axis=0), 48)
        weights = np.divide(np.power(MSPE, -2), np.sum(np.power(MSPE,-2)))
        preds.insert(i, np.sum(np.multiply(learner_predictions[i],weights)))
        ensemble_weights.append(weights)

y_actual = df[2880:20160, -1]
residual = np.abs(y_actual-preds)
ensemble_weights = np.concatenate(ensemble_weights).reshape(len(learner_predictions),5)

# =========================
#ES Yesterday Ensemble Evaluation
# =========================
RMSE = metrics.mean_squared_error(y_actual, preds, squared = False)
MAE = metrics.mean_absolute_error(y_actual, preds)

print("ES - Yesterday, RMSE: " + str(RMSE))
print("ES - Yesterday, MAE: " + str(MAE))

# =========================
#Plottting Forecasts
# =========================
plt.figure(figsize=(12, 6))
sns.set_style("darkgrid")
plt.title('ES - Yesterday: First Five Days').set_fontsize(20)
plt.plot(y_actual[:240],'--', label="Actual Value", color='#8c8488')
plt.plot(preds[:240], label="ES - Yesterday", color='#0a4236')
plt.plot(residual[:240], label="Residual", color='red', alpha=0.2)
plt.legend(loc=1)
plt.ylabel('System Sell Price (£/MWh)').set_fontsize(16)
plt.xlabel('Settlement Period').set_fontsize(16)
plt.show(block=False)

# =========================
#Plottting Ensemble Weights
# =========================
plt.figure(figsize=(12, 6))
sns.set_style("darkgrid")
plt.title('ES - Yesterday: Model Weights').set_fontsize(20)
y_axis_labels = ['RFR', 'SVR','FFNN','GRU','LSTM'] # labels for y-axis
g = sns.heatmap(ensemble_weights.transpose(), yticklabels=y_axis_labels, cmap="PuBuGn")
plt.show(block=False)

# =========================
#Export Test Predections and Residuals
# =========================
test_preds = pd.read_csv(r"C:\Users\Owen\OneDrive - Imperial College London\Research Project\Thesis\Final\2_Models\3_Ensemble\ensemble_predictions.csv", index_col=[0])
test_preds.iloc[:,1] = preds
test_preds.to_csv(r"C:\Users\Owen\OneDrive - Imperial College London\Research Project\Thesis\Final\2_Models\3_Ensemble\ensemble_predictions.csv")

test_preds = pd.read_csv(r"C:\Users\Owen\OneDrive - Imperial College London\Research Project\Thesis\Final\2_Models\3_Ensemble\ensemble_errors.csv", index_col=[0])
test_preds.iloc[:,1] = residual
test_preds.to_csv(r"C:\Users\Owen\OneDrive - Imperial College London\Research Project\Thesis\Final\2_Models\3_Ensemble\ensemble_errors.csv")
# %%
