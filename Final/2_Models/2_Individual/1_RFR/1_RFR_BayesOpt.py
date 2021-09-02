#%%
# =========================
#Bayesian Optimisation for the RFR
# =========================
import pandas as pd
import numpy as np
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK, space_eval
from hyperopt.pyll import scope
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.notebook import tqdm
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
space = {'max_depth': scope.int(hp.quniform('max_depth', 10,100,10)),
        'min_samples_leaf': scope.int(hp.quniform('min_samples_leaf', 1,12,1)),
        'min_samples_split': scope.int(hp.quniform('min_samples_split', 2,12,1)),
        'n_estimators': scope.int(hp.quniform('n_estimators', 25,200,25)),
        }

# =========================
#Definde RFR objective function for Bayesian Optimisation
# =========================
def objective(space):
    RFR = RandomForestRegressor(max_depth = space['max_depth'],
                                min_samples_leaf = space['min_samples_leaf'],
                                min_samples_split =  space['min_samples_split'],
                                n_estimators = space['n_estimators'],
                                max_features='sqrt')

    #train and predict 
    RFR.fit(X_train, y_train)
    val_pred = RFR.predict(X_val)

    #return RMSE metric
    accuracy = metrics.mean_squared_error(y_val, val_pred, squared=False)

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

#Extraxt best HP
best['max_depth'] = best['max_depth'].astype(int)
best['n_estimators'] = best['n_estimators'].astype(int)
best['min_samples_leaf'] = best['min_samples_leaf'].astype(int)
best['min_samples_split'] = best['min_samples_split'].astype(int)

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
hp_best['max_depth'] = hp_best['max_depth']
hp_best['n_estimators'] = hp_best['n_estimators']
#scatterplot
plt.figure(figsize=(8, 5))
sns.set_style("darkgrid")
sns.scatterplot(data=trials_df, x="trial number", y="RMSE", color="#1e4f5e")
sns.scatterplot(data=hp_best, x="trial number", y="RMSE", color="#459981", label="Best")

# =========================
#Define and Save Optimal Model
# =========================
RFR_test = RandomForestRegressor(**best, max_features='sqrt')

with open('RFR_pickle', 'wb') as f:
    pickle.dump(RFR_test, f)


# %%
