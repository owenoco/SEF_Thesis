#%%
# =========================
#Calculating Monethly Errors for Individual Models
# =========================
import pandas as pd
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns


# =========================
#Import Data
# =========================
df = pd.read_csv(r"C:\Users\Owen\OneDrive - Imperial College London\Research Project\Thesis\Data Collection\feature_engineered_data_2.csv", index_col=[0])
data = df.iloc[1440:20160, :] #13 months
data = np.asarray(data)

#Individual Model Predictions (currently empty)
indiv_preds = pd.read_csv(r"C:\Users\Owen\OneDrive - Imperial College London\Research Project\Thesis\Final\2_Models\2_Individual\Learner_Predictions\12_month_learner_predictions.csv", index_col=[0])
indiv_preds = np.asarray(indiv_preds)

#Individual Model Errors (currently empty)
monthly_mae = pd.read_csv(r"C:\Users\Owen\OneDrive - Imperial College London\Research Project\Thesis\Final\2_Models\2_Individual\Learner_Errors\monthly_mae.csv", index_col=[0])
monthly_rmse = pd.read_csv(r"C:\Users\Owen\OneDrive - Imperial College London\Research Project\Thesis\Final\2_Models\2_Individual\Learner_Errors\monthly_rmse.csv", index_col=[0])

y_actual = data[1440:, -1]

# =========================
#Calculate Monthly MAE and RMSE of Individual Models
# =========================
for a in range(0,5):

    for i in range(monthly_mae.shape[0]):
        month_preds = indiv_preds[i*1440:(i+1)*1440, a]
        month_actual = y_actual[i*1440:(i+1)*1440]

        monthly_mae.iloc[i,a] = metrics.mean_absolute_error(month_actual, month_preds)
        monthly_rmse.iloc[i,a] = metrics.mean_squared_error(month_actual, month_preds, squared = False)


# =========================
#Plot Monthly MAE and RMSE of Individual Models
# =========================
#MAE
plt.figure(figsize=(12,5))
sns.set_style("darkgrid")
plt.title('Monthly MAE').set_fontsize(20)
plt.plot(monthly_mae.iloc[:,0], label='RFR', color='#bcbcc3', marker="1")
plt.plot(monthly_mae.iloc[:,1], label='SVR', color='#66897d', marker="1")
plt.plot(monthly_mae.iloc[:,2], label='FFNN', color='#a5daaf', marker="1")
plt.plot(monthly_mae.iloc[:,3], label='GRU', color='#439c95', marker="1")
plt.plot(monthly_mae.iloc[:,4], label='LSTM', color='#1f4e60', marker="1")
plt.ylabel('Error').set_fontsize(16)
plt.xlabel('Month').set_fontsize(16)
plt.xticks(np.arange(0,12), ['Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'Jan', 'Feb'])
plt.ylim(0,30)
plt.legend()

#RMSE
plt.figure(figsize=(12,5))
sns.set_style("darkgrid")
plt.title('Monthly RMSE').set_fontsize(20)
plt.plot(monthly_rmse.iloc[:,0], label='RFR', color='#bcbcc3', marker="1")
plt.plot(monthly_rmse.iloc[:,1], label='SVR', color='#66897d', marker="1")
plt.plot(monthly_rmse.iloc[:,2], label='FFNN', color='#a5daaf', marker="1")
plt.plot(monthly_rmse.iloc[:,3], label='GRU', color='#439c95', marker="1")
plt.plot(monthly_rmse.iloc[:,4], label='LSTM', color='#1f4e60', marker="1")
plt.ylabel('Error').set_fontsize(16)
plt.xlabel('Month').set_fontsize(16)
plt.xticks(np.arange(0,12), ['Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'Jan', 'Feb'])
plt.ylim(0,30)
plt.legend()

# =========================
#Export Monthly MAE and RMSE of Individual Models
# =========================
monthly_mae.to_csv(r"C:\Users\Owen\OneDrive - Imperial College London\Research Project\Thesis\Final\2_Models\2_Individual\Learner_Errors\monthly_mae.csv")
monthly_rmse.to_csv(r"C:\Users\Owen\OneDrive - Imperial College London\Research Project\Thesis\Final\2_Models\2_Individual\Learner_Errors\monthly_rmse.csv")
# %%
