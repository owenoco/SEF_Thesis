#%%
# =========================
#RFR Twelve Month Evaluation
# =========================
import matplotlib
import pandas as pd
import numpy as np
from torch.utils.data.sampler import BatchSampler
from tqdm.notebook import tqdm
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
import itertools as it
import pickle


# =========================
#Load Dataset
# =========================
df = pd.read_csv(r"C:\Users\Owen\OneDrive - Imperial College London\Research Project\Thesis\Data Collection\feature_engineered_data_2.csv", index_col=[0])
data = df.iloc[1440:20160, :] #13 months
data = np.asarray(data)

# =========================
#Define Sliding Window Function and Dataset
# =========================
step_size = 1
def moving_window(x, length, step=step_size):
    streams = it.tee(x, length)
    return zip(*[it.islice(stream, i, None, step) for stream, i in zip(streams, it.count(step=step))])

#Sliding window params
window_length = 1441 #1 Months
pred_window = len(data)-window_length #12 Months

test_set = list(moving_window(data, window_length))
test_set = np.asarray(test_set)

# =========================
#RFR Tetsing
# =========================
y_test_pred = []                        
for i in tqdm(range(0, len(test_set))):

    X_f_train = test_set[i, :-1, :-1]
    y_f_train = test_set[i, :-1, -1]

    X_test = test_set[i, -1, :-1]
    X_test = X_test.reshape(1,22)
    
    RFR_test = pickle.load(open('RFR_pickle', 'rb'))
    
    #train and predict 
    RFR_test.fit(X_f_train, y_f_train)
    test_pred = RFR_test.predict(X_test)
    y_test_pred.append(test_pred)

y_test_pred = np.concatenate((y_test_pred))
y_test_list = test_set[-(pred_window+1):, -1, -1]
residual = np.abs(np.asarray(y_test_list)-y_test_pred)

# =========================
#RFR Evaluation
# =========================
RFR_RMSE = metrics.mean_squared_error(y_test_list, y_test_pred, squared = False)
RFR_MAE = metrics.mean_absolute_error(y_test_list, y_test_pred)

print("RFR RMSE: " + str(RFR_RMSE))
print("RFR MAE: " + str(RFR_MAE))

# =========================
#Plotting Forecasts
# =========================
plt.figure(figsize=(12, 6))
sns.set_style("darkgrid")
plt.title('RFR Model - First Five Days').set_fontsize(20)
plt.plot(y_test_list[:240],'--', label="Actual Value", color='#8c8488')
plt.plot(y_test_pred[:240], label="RFR", color='#0a4236')
plt.plot(residual[:240], label="Residual", color='red', alpha=0.2)
plt.legend(loc=1)
plt.ylabel('System Sell Price (£/MWh)').set_fontsize(16)
plt.xlabel('Settlement Period').set_fontsize(16)

# =========================
#Export Test Predections and Residuals
# =========================
test_preds = pd.read_csv(r"C:\Users\Owen\OneDrive - Imperial College London\Research Project\Thesis\Final\2_Models\2_Individual\Learner_Predictions\12_month_learner_predictions.csv", index_col=[0])
test_preds.iloc[:,0] = y_test_pred
test_preds.to_csv(r"C:\Users\Owen\OneDrive - Imperial College London\Research Project\Thesis\Final\2_Models\2_Individual\Learner_Predictions\12_month_learner_predictions.csv")

indiv_errors = pd.read_csv(r"C:\Users\Owen\OneDrive - Imperial College London\Research Project\Thesis\Final\2_Models\2_Individual\Learner_Errors\12_month_learner_errors.csv", index_col=[0])
indiv_errors.iloc[:,0] = residual
indiv_errors.to_csv(r"C:\Users\Owen\OneDrive - Imperial College London\Research Project\Thesis\Final\2_Models\2_Individual\Learner_Errors\12_month_learner_errors.csv")
# %%
