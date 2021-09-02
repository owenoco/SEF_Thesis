#%%
import pandas as pd
from epftoolbox.evaluation import DM, plot_multivariate_DM_test

df = pd.read_csv(r"C:\Users\Owen\OneDrive - Imperial College London\Research Project\Thesis\Data Collection\feature_engineered_data_2.csv", index_col=[0])
actual_price = df.iloc[2880:20160, -1]

indiv_preds = pd.read_csv(r"C:\Users\Owen\OneDrive - Imperial College London\Research Project\Thesis\Final\2_Models\2_Individual\Learner_Predictions\12_month_learner_predictions.csv", index_col=[0])
ensemble_preds = pd.read_csv(r"C:\Users\Owen\OneDrive - Imperial College London\Research Project\Thesis\Final\2_Models\3_Ensemble\ensemble_predictions.csv", index_col=[0])

preds = pd.concat((indiv_preds, ensemble_preds), axis=1)

plot_multivariate_DM_test(real_price=actual_price, forecasts=preds)


# %%
