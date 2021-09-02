#%%
# =========================
#LSTM Twelve Month Evaluation
# =========================
from hyperopt.pyll_utils import hp_choice
import pandas as pd
import numpy as np
from tqdm.notebook import tqdm
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
import itertools as it

import torch
import torch.nn as nn 
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import BatchSampler

from hyperopt import hp, fmin, tpe, Trials, STATUS_OK, space_eval
from hyperopt.pyll import scope, stochastic

# =========================
#Define Dataset initialization class
# =========================
class Dataset(Dataset):
    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data
    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]
    def __len__ (self):
        return len(self.X_data)

# =========================
#Define Hyperparameters
# =========================
input_size = 22
output_size = 1
epochs = 100
batch_size = 256
learning_rate = 0.01
momentum = 0.1
n_hidden = 64
criterion = nn.MSELoss()

#MLP Architecture
class MLP(nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super().__init__()
        self.fc1 = nn.Linear(n_feature, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_hidden)
        self.fc3 = nn.Linear(n_hidden, n_output)
        self.dropout = nn.Dropout(0.5)
        self.apply(self.init_weights)
    
    def init_weights(self, m):
        if type(m) == nn.Linear:
            torch.nn.init.kaiming_uniform_(m.weight)
            m.bias.data.fill_(1)

    def forward(self, x):
        x = self.dropout(x)
        x = F.leaky_relu(self.fc1(x))
        x = self.dropout(x)
        x = F.leaky_relu(self.fc2(x))
        x = self.fc3(x)
        return x


# =========================
#Define Sliding Window Function
# =========================
step_size = 1
def moving_window(x, length, step=step_size):
    streams = it.tee(x, length)
    return zip(*[it.islice(stream, i, None, step) for stream, i in zip(streams, it.count(step=step))])


# =========================
#Load Dataset
# =========================

df = pd.read_csv(r"C:\Users\Owen\OneDrive - Imperial College London\Research Project\Thesis\Data Collection\feature_engineered_data_2.csv", index_col=[0])
data = df.iloc[1440:20160, :] #13 months
data = np.asarray(data)


# =========================
#FFNN Tetsing
# =========================
#Sliding window params
window_length = 1441 #1 Months
pred_window = len(data)-window_length #12 Months

test_set = list(moving_window(data, window_length))
test_set = np.asarray(test_set)

y_pred_list = []                        
for i in tqdm(range(0, len(test_set))):

    X_train = test_set[i, :-1, :-1]
    y_train = test_set[i, :-1, -1]

    X_test = test_set[i, -1, :-1]
    X_test = X_test.reshape(1,22)

    scaler = MinMaxScaler(feature_range=(-1,1)) 
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test.reshape(1, -1))
    
    X_train, y_train = np.array(X_train), np.array(y_train)
   
    X_test, np.array(X_test)
    X_test = torch.from_numpy(X_test).float()
    
    train_dataset = Dataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).float())
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size)


    model=MLP(input_size,n_hidden,output_size)
    optimizer = optim.RMSprop(model.parameters(), lr=learning_rate, momentum=momentum)

    for e in range(0, epochs):
        model.train()
        for X_train_batch, y_train_batch in train_loader:
            X_train_batch = X_train_batch
            optimizer.zero_grad()
            y_train_pred = model(X_train_batch)
            train_loss = criterion(y_train_pred, y_train_batch.unsqueeze(1))
            train_loss.backward()
            optimizer.step()

    with torch.no_grad():
        y_test_pred = model(X_test)
        y_pred_list.append(y_test_pred.item())

y_actual = test_set[-(pred_window+1):, -1, -1]
residual = np.abs(np.asarray(y_actual)-y_pred_list)

# =========================
#FFNN Evaluation
# =========================
MLP_RMSE = metrics.mean_squared_error(y_actual, y_pred_list, squared = False)
MLP_MAE = metrics.mean_absolute_error(y_actual, y_pred_list)

print("MLP RMSE: " + str(MLP_RMSE))
print("MLP MAE: " + str(MLP_MAE))

# =========================
#Plotting Forecasts
# =========================
plt.figure(figsize=(12, 6))
sns.set_style("darkgrid")
plt.title('MLP - First Five Days').set_fontsize(20)
plt.plot(y_actual[:240],'--', label="Actual Value", color='#8c8488')
plt.plot(y_pred_list[:240], label="MLP", color='#0a4236')
plt.plot(residual[:240], label="Residual", color='red', alpha=0.2)
plt.legend(loc=1)
plt.ylabel('System Sell Price (£/MWh)').set_fontsize(16)
plt.xlabel('Settlement Period').set_fontsize(16)
plt.show(block=False)

# =========================
#Export Test Predections and Residuals
# =========================
test_preds = pd.read_csv(r"C:\Users\Owen\OneDrive - Imperial College London\Research Project\Thesis\Final\2_Models\2_Individual\Learner_Predictions\12_month_learner_predictions.csv", index_col=[0])
test_preds.iloc[:,2] = y_pred_list
test_preds.to_csv(r"C:\Users\Owen\OneDrive - Imperial College London\Research Project\Thesis\Final\2_Models\2_Individual\Learner_Predictions\12_month_learner_predictions.csv")

indiv_errors = pd.read_csv(r"C:\Users\Owen\OneDrive - Imperial College London\Research Project\Thesis\Final\2_Models\2_Individual\Learner_Errors\12_month_learner_errors.csv", index_col=[0])
indiv_errors.iloc[:,2] = residual
indiv_errors.to_csv(r"C:\Users\Owen\OneDrive - Imperial College London\Research Project\Thesis\Final\2_Models\2_Individual\Learner_Errors\12_month_learner_errors.csv")

# %%
