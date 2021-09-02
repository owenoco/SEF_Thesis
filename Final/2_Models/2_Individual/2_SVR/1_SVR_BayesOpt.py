#%%
# =========================
#Bayesian Optimisation for the SVR
# =========================
import pandas as pd
import numpy as np
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK, space_eval
from hyperopt.pyll import scope, stochastic
from plotly import express as px
from plotly import graph_objects as go
from plotly import offline as pyo
from sklearn.svm import SVR
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
import itertools as it
from tqdm.notebook import tqdm
from sklearn.preprocessing import MinMaxScaler
import pickle


# =========================
#Import First 6 weeks of Data
# =========================
df = pd.read_csv(r"C:\Users\Owen\OneDrive - Imperial College London\Research Project\Thesis\Data Collection\feature_engineered_data_2.csv", index_col=[0])
data = df.iloc[:2064, :] 
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

# =========================
# Define Bayesian Optimisation search space
# =========================
space = {'C': hp.choice('C', [1,10,25,50,75,100]),
        'epsilon': hp.choice('epsilon', [1e-3,1e-2,1e-1,1]),
        'gamma': hp.choice('gamma', ['scale','auto']),
        }


# =========================
#Definde SVR objective function for Bayesian Optimisation
# =========================
def objective(space):
    model = SVR(C = space['C'],
                epsilon=space['epsilon'],
                gamma = space['gamma'],)
    #train and predict 
    model.fit(X_train, y_train)
    pred = model.predict(X_val)

    #train and predict 
    model.fit(X_train, y_train)
    pred = model.predict(X_val)

    #return RMSE metric
    accuracy = metrics.mean_squared_error(y_val, pred, squared=False)

    return {'loss': accuracy, "status":STATUS_OK}

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
# #extracting best hyperparameters
hp_best = trials_df[trials_df.RMSE == trials_df.RMSE.min()].copy()
#scatterplot
plt.figure(figsize=(8, 5))
sns.set_style("darkgrid")
sns.scatterplot(data=trials_df, x="trial number", y="RMSE", color="#1e4f5e")
sns.scatterplot(data=hp_best, x="trial number", y="RMSE", color="#459981", label="Best")

# =========================
#Define and Save Optimal Model
# =========================
SVR_test = SVR(**best_hp)

with open('SVR_pickle', 'wb') as f:
    pickle.dump(SVR_test, f)

# %%
