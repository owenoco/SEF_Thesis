# =========================
#Feature Engineering
# =========================
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
# =========================
#Import Pre-Processed Data
# =========================
df = pd.read_csv(r"C:\Users\Owen\OneDrive - Imperial College London\Research Project\Thesis\Data Collection\pre_processed_data_2.csv", parse_dates=["Date"])
# =========================
#Grouping
# =========================
df["DA_Renewables"] = df["day_ahead_solar"]+df["day_ahead_wind_offshore"]+df["day_ahead_wind_onshore"]
df["total_interconnectors"] = df["interconnector_britnes"]+df["interconnector_ewic"]+df["interconnector_ifa"]+df["interconnector_moyle"]+df["interconnector_nemo"]
df["total_ren_gen"] = df["solar"]+df["wind_outturn"]
# =========================
#Encoding
# =========================
df["SP_sin"] = np.sin(2 * np.pi * df['SP']/48.0)
DOW = df['Date'].copy().dt.dayofweek
df["DOW_sin"] = np.sin(2 * np.pi * DOW/7)
df["Target"] = df["system_sell_price_2"]
# =========================
#Dropping Redundent Features
# =========================
df = df.drop(columns=['Unnamed: 0',
            'Date',
            'SP',
            'day_ahead_solar',
            'day_ahead_wind_offshore',
            'day_ahead_wind_onshore',
            'initial_wind_forecast',
            'lolp_2hour',
            'lolp_4hour',
            'lolp_8hour',
            'interconnector_britnes',
            'interconnector_ewic',
            'interconnector_ifa',
            'interconnector_moyle',
            'interconnector_nemo',
            'solar',
            'wind_offshore',
            'wind_onshore',
            'wind_outturn',
            'system_sell_price_2'])

# =========================
#Pearson and Spearman Correlation Analysis
# =========================
correlations_p = df.corr(method='pearson')
Imbalance_Pearson = np.abs(correlations_p["Target"])

correlations_s = df.corr(method='spearman')
Imbalance_Spearman  = np.abs(correlations_s["Target"])

Imbalance_Corr = pd.concat([Imbalance_Pearson, Imbalance_Spearman], axis=1)
Imbalance_Corr = Imbalance_Corr.drop(Imbalance_Corr['Target'])

Imbalance_Corr.columns = ['Pearson', 'Spearman']
Corr_plot = Imbalance_Corr.copy()
Corr_plot['Feature'] = Corr_plot.index
Corr_plot
Corr_plot = pd.melt(Corr_plot, id_vars="Feature", var_name="Method", value_name="Correlation Value")

# =========================
#Correlation Plot
# =========================
plt.figure(figsize=(12,8))
sns.set_style("darkgrid")
sns.set(font_scale = 2)
cat_plot = sns.catplot(x='Feature', y='Correlation Value', hue='Method', data=Corr_plot, kind='bar', height=10, aspect=2, palette="crest", legend=False)
cat_plot.set(ylim=(0, 0.6))
cat_plot.set_xticklabels(rotation=85, size=18)
plt.title("Correlation Analysis").set_fontsize(30)
plt.xlabel("Feature").set_fontsize(26)
plt.ylabel("Correlation Value").set_fontsize(26)
plt.legend(loc='upper right')
plt.axhline(0.05, ls='--',color='red')

# =========================
#Export the Final Dataset
# =========================
feature_engineered_data = df.iloc[336:].copy().reset_index().drop(columns=['index'])
feature_engineered_data.to_csv(r"C:\Users\Owen\OneDrive - Imperial College London\Research Project\Thesis\Data Collection\feature_engineered_data_2.csv")
