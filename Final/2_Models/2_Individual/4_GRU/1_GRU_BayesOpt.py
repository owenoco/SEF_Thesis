#%%
# =========================
#Bayesian Optimisation for the GRU
# =========================
import pandas as pd
import numpy as np
from tqdm.notebook import tqdm
from sklearn.preprocessing import MinMaxScaler
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
#Define GRU Model Architecture
# =========================
class GRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers):
        super(GRU, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.gru = nn.GRU(input_dim, hidden_dim, n_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.apply(self.init_weights)

    def init_weights(self, m):
        if type(m) == nn.Linear:
            torch.nn.init.kaiming_uniform_(m.weight)
            m.bias.data.fill_(1)

    def forward(self, x):
        h0 = torch.zeros(self.n_layers, x.size(0), self.hidden_dim).requires_grad_()
        x, hn = self.gru(x, (h0.detach()))
        x = self.fc(x[:,-1,:])
        return x



# =========================
#Import First 6 weeks of Data
# =========================
df = pd.read_csv(r"C:\Users\Owen\OneDrive - Imperial College London\Research Project\Thesis\Data Collection\feature_engineered_data_2.csv", index_col=[0])
data = df.iloc[:2064, :]  #4 months
data = np.asarray(data)
# =========================
#Split Data into train/validate
# =========================
#30 Days
X_train = data[:1440, :-1]
y_train = data[:1440, -1]
#13 Days
X_val = data[1440:2064, :-1]
y_val = data[1440:2064, -1]

#Normalize dataset
scaler = MinMaxScaler(feature_range=(-1, 1))
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

# =========================
#Define Presetermined HyperParameters
# =========================
input_size = 22
output_size = 1
epochs = 100
batch_size = 256

# =========================
# Define Bayesian Optimisation search space
# =========================
space = {'hidden_size': hp.choice('hidden_size', [16,32,64,128,256]),
        'lr': hp.choice('lr', [0.0001,0.001,0.01,0.1]),
        'momentum': hp.quniform('momentum', 0.1,1,0.1),
        'sequence_length': scope.int(hp.quniform('sequence_length', 1,48,4)),
        }

criterion = nn.MSELoss()

# =========================
#Definde FFNN objective function for Bayesian Optimisation
# =========================
def objective(space):

    X_train = data[:1440, :-1]
    y_train = data[:1440, -1]
    X_val = data[1440:2880, :-1]
    y_val = data[1440:2880, -1]
    
    # =========================
    #Organsiing Data to accomadate RNN sequence length
    # =========================
    y_train = np.array(y_train[space['sequence_length']:]).astype(np.float32)
    X_train_rnn = np.empty([X_train.shape[0]-space['sequence_length'], space['sequence_length'], input_size])
    for a in range(X_train_rnn.shape[0]):
        X_train_rnn[a] = X_train[a:a+space['sequence_length']]

    y_val = np.array(y_val[space['sequence_length']:]).astype(np.float32)
    X_val_rnn = np.empty([X_val.shape[0]-space['sequence_length'], space['sequence_length'], input_size])
    for b in range(X_val_rnn.shape[0]):
        X_val_rnn[b] = X_val[b:b+space['sequence_length']]
    # =========================
    #Insert into Dataloader for Pytorch
    # =========================
    train_dataset = Dataset(torch.from_numpy(X_train_rnn).float(), torch.from_numpy(y_train).float())
    val_dataset = Dataset(torch.from_numpy(X_val_rnn).float(), torch.from_numpy(y_val).float())
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size)
    
    model = GRU(input_dim=input_size, 
                  hidden_dim=space['hidden_size'],
                  n_layers=1,
                  output_dim=output_size)
    optimizer = optim.RMSprop(model.parameters(), lr=space['lr'], momentum=space['momentum'])

    y_val_pred_list = []
    for e in range(0, epochs):
        model.train()
        for X_train_batch, y_train_batch in train_loader:
            optimizer.zero_grad()
            y_train_pred = model(X_train_batch)
            train_loss = criterion(y_train_pred, y_train_batch.unsqueeze(1))
            train_loss.backward()
            optimizer.step()

    with torch.no_grad():
        for X_val_batch, _ in val_loader:
            y_val_pred = model(X_val_batch)
            y_val_pred_list.append(y_val_pred.cpu().detach().numpy())

    y_val_pred_list = np.concatenate(y_val_pred_list, axis=0 )
    RMSE = metrics.mean_squared_error(y_val, y_val_pred_list, squared = False)
    return RMSE

# =========================
#HyperOpt Run - Minimize Objective Function with 100 iterations
# =========================
trials = Trials()
best = fmin(fn = objective,
            space = space,
            algo = tpe.suggest,
            max_evals = 100,
            trials = trials,
            )
best_hp = space_eval(space, best)
print('Best Hyperparameters:' + str(best_hp))

# =========================
# Extract Bayesian Optimisation results into a dataframe
# =========================
# fill in `np.nan` when a particular hyperparameter is not relevant to a particular trial
def unpack(x):
    if x:
        return x[0]
    return np.nan
# Turn each trial into a series and then stack those series together as a dataframe.
trials_df = pd.DataFrame([pd.Series(t["misc"]["vals"]).apply(unpack) for t in trials])
# Add loss values and trial numbers to DF
trials_df["RMSE"] = [t["result"]["loss"] for t in trials]
trials_df["trial number"] = trials_df.index
#extracting best hyperparameters
hp_best = trials_df[trials_df.RMSE == trials_df.RMSE.min()].copy()

#scatterplot
plt.figure(figsize=(8, 5))
sns.set_style("darkgrid")
sns.scatterplot(data=trials_df, x="trial number", y="RMSE", color="#1e4f5e")
sns.scatterplot(data=hp_best, x="trial number", y="RMSE", color="#459981", label="Best")

# %%
