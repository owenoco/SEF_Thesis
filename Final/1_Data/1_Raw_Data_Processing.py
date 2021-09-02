# =========================
#Raw Data Pre-Processing
# =========================
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import KNNImputer
import missingno as msno

# =========================
#Import raw data and drop final day values (many missing)
# =========================
df = pd.read_csv(r"C:\Users\Owen\OneDrive - Imperial College London\Research Project\Thesis\Data Collection\Raw Data\data.csv")
df.drop(df.tail(43).index,inplace=True)
imbalance_price = df['system_sell_price'].copy()
# =========================
#Visualise Target Variable
# =========================
plt.figure(figsize=(16,8))
sns.set_style("darkgrid")
plt.plot(df['system_sell_price'], color='#1e4f60')
plt.title('System Sell Price - Entire Dataset').set_fontsize(20)
plt.ylabel('System Sell Price (£/MWh)').set_fontsize(16)
plt.xlabel('Settlement Period').set_fontsize(16)
imbalance_price.describe()
plt.figure(figsize=(16,8))
msno.matrix(df)
# =========================
#Imputing NaNs from two SPs
# =========================
df1 = df.iloc[:21790]
df2 = df.iloc[21790:].shift(periods=2)
df = pd.concat([df1, df2])
# =========================
#Adding Formatted Calander Features
# =========================
sp = np.arange(1,49,1)
settlement_period = np.tile(sp, 850)
settlement_period = pd.Series(settlement_period, name='SP')
dates = pd.to_datetime(df['date_time'].copy())
dates = dates.dt.date
dates = pd.Series(dates, name='Date')
dates = pd.to_datetime(dates)
df = pd.concat([dates, settlement_period, df], axis=1)
# =========================
#Drop unnecessary date features
# =========================
df = df.drop(columns=['Period Start', 'date_time'])

# =========================
#shift 'live' data points
# =========================
shifted_features = df[['buy_price_adjustment',
            'initial_transmission_system_demand_outturn',
            'interconnector_britnes',
            'interconnector_ewic',
            'interconnector_ifa',
            'interconnector_moyle',
            'interconnector_nemo',
            'market_index_apx_price',
            'market_index_apx_volume',
            'net_imbalance_volume',
            'reserve_scarcity_price',
            'solar',
            'system_sell_price',
            'total_generation',
            'wind_offshore',
            'Bid TOS',
            'Offer TOS']].copy().shift(periods=1)

target = df['system_sell_price'].copy().shift(periods=-3)

df = df.drop(columns=['buy_price_adjustment',
            'initial_transmission_system_demand_outturn',
            'interconnector_britnes',
            'interconnector_ewic',
            'interconnector_ifa',
            'interconnector_moyle',
            'interconnector_nemo',
            'market_index_apx_price',
            'market_index_apx_volume',
            'net_imbalance_volume',
            'reserve_scarcity_price',
            'solar',
            'system_sell_price',
            'total_generation',
            'wind_offshore',
            'Bid TOS',
            'Offer TOS'])
df = pd.concat([df, shifted_features, target], axis=1)

# =========================
#The target system sell price and most recent system sell price have the same name, this makes each one unique
# =========================
cols = []
count = 1
for column in df.columns:
    if column == 'system_sell_price':
        cols.append(f'system_sell_price_{count}')
        count+=1
        continue
    cols.append(column)
df.columns = cols

# =========================
#Impute missing values
# =========================
df['initial_wind_forecast'] = df['initial_wind_forecast'].fillna(method='ffill')
df['latest_wind_forecast'] = df['latest_wind_forecast'].fillna(method='ffill')
df['reserve_scarcity_price'] = df['reserve_scarcity_price'].fillna(0)
df['interconnector_nemo'] = df['interconnector_nemo'].fillna(0)
df['Date'] = df['Date'].fillna(method='ffill')
df['SP'] = df['SP'].fillna(1)
df_imputed = df.interpolate(method='linear', axis=0)
df_imputed = df_imputed.drop(df_imputed.index[0])
missing = np.asanyarray(np.isnan(df['Offer TOS'].iloc[:300]))
missing_index = df_imputed['Offer TOS'].iloc[np.where(missing)]


# =========================
#Plot of Interpolated Values
# =========================
fig = plt.figure(figsize=(10,5))
sns.set_style("darkgrid")
plt.title('BM - Highest Offer', fontsize=20)
plt.plot(df_imputed['Offer TOS'].iloc[:300], '--', color='red', alpha=0.4)
plt.scatter(x=np.where(missing), y=missing_index, color='red',  marker="1",label='Interpolated')
plt.plot(df['Offer TOS'].iloc[:300], color='#29677c')
plt.xlabel('Settlement Period', fontsize=16)
plt.ylabel('Price (£/MWh)', fontsize=16)
plt.legend()
plt.show()

# =========================
#Final check to ensure no missing values
# =========================
df_imputed.isna().sum()

# =========================
#Export Dataset
# =========================
pre_processed_data = df_imputed.copy()
pre_processed_data.to_csv(r"C:\Users\Owen\OneDrive - Imperial College London\Research Project\Thesis\Data Collection\pre_processed_data_2.csv")
