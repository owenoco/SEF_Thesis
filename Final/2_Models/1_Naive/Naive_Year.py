#%%
# ==========================
# Naive Models:
## Mean Value
## Previous SP
## Yesterday SP
# ==========================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import metrics
import seaborn as sns

# =========================
# Import Data
# =========================
data = pd.read_csv(r"C:\Users\Owen\OneDrive - Imperial College London\Research Project\Thesis\Data Collection\feature_engineered_data_2.csv", index_col=[0])
y = np.asarray(data["Target"].iloc[1440:20160])
y_fit = data["Target"].iloc[2880:20160].copy()
y_fit = y_fit.reset_index().drop(columns='index')


# =========================
#Naive Model - Latest Imbalance Price 
# =========================
shift_data_1 = data["system_sell_price_1"].copy()
y_prev_sp = shift_data_1.iloc[2880:20160]
y_prev_sp = y_prev_sp.reset_index().drop(columns='index')

#Error Metrics
PrevSP_RMSE = metrics.mean_squared_error(y_fit, y_prev_sp, squared = False)
PrevSP_MAE = metrics.mean_absolute_error(y_fit, y_prev_sp)

print("Naive Previous SP RMSE: " + str(PrevSP_RMSE))
print("Naive Previous SP MAE: " + str(PrevSP_MAE))

# =========================
#Naive Model - Yesterday SP 
# =========================
 
shift_data_48 = data["Target"].copy().shift(periods=48)
y_yesterday_sp = shift_data_48.iloc[2880:20160]
y_yesterday_sp = y_yesterday_sp.reset_index().drop(columns='index')

#Error Metrics
YST_SP_RMSE = metrics.mean_squared_error(y_fit, y_yesterday_sp, squared = False)
YST_SP_MAE = metrics.mean_absolute_error(y_fit, y_yesterday_sp)

print("Naive Previous Day RMSE: " + str(YST_SP_RMSE))
print("Naive Previous Day MAE: " + str(YST_SP_MAE))

# =========================
#Naive Model - Mean  
# =========================
y_pred_mean = []
for i in range(0, len(y_fit)):
    y_pred_mean.append(y[i:i+1437].mean())

y_pred_mean = pd.DataFrame(data=y_pred_mean)

# =========================
#Error Metrics
# =========================
Mean_RMSE = metrics.mean_squared_error(y_fit, y_pred_mean, squared = False)
Mean_MAE = metrics.mean_absolute_error(y_fit, y_pred_mean)

print("Naive Mean RMSE: " + str(Mean_RMSE))
print("Naive Mean MAE: " + str(Mean_MAE))

# =========================
#Absolute Error DF
# =========================

AE = {'mean': y_fit['Target'].sub(y_pred_mean.iloc[:, 0]).abs(),
'prev_sp': y_fit['Target'].sub(y_prev_sp.iloc[:, 0]).abs(),
'prev_day': y_fit['Target'].sub(y_yesterday_sp.iloc[:, 0]).abs(),
}
AbEr = pd.DataFrame(AE)

# =========================
#Create Plot
# =========================
fig, ax = plt.subplots(3, 1, figsize=(12, 8))
sns.set_style("darkgrid")
fig.suptitle('Naive Method Comparison - First Five Days', position=(0.5, 0.92)).set_fontsize(20)

ax[0].plot(y_fit[:240],'--', color='#8c8488')
ax[0].plot(y_prev_sp[:240], label="Naive - Latest SP", color='#0a4236')
ax[0].plot(AbEr['prev_sp'][:240], color='red', alpha=0.2)
ax[0].legend(loc=1)

ax[1].plot(y_fit[:240],'--', label="Actual Price", color='#8c8488')
ax[1].plot(AbEr['prev_day'][:240], label="Residual", color='red', alpha=0.2)
ax[1].plot(y_yesterday_sp[:240], label="Naive - Previous Day SP", color='#0a4236')
ax[1].legend(loc=1)

ax[2].plot(y_fit[:240],'--', color='#8c8488')
ax[2].plot(y_pred_mean[:240], color='#0a4236', label="Naive - Mean")
ax[2].plot(AbEr['mean'][:240], color='red',  alpha=0.2)
ax[2].legend(loc=1)

ax[1].set_ylabel('System Price (£/MWh)').set_fontsize(16)
ax[2].set_xlabel('Settlement Period').set_fontsize(16)

plt.show()


# %%
