import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Load and preprocess your data
data = pd.read_csv('data_daily.csv', parse_dates=['# Date'], index_col='# Date')

# ACF and PACF plots to determine p and q
time_series = data['Receipt_Count']

# Plot the Autocorrelation Function (ACF)
plt.figure(figsize=(12,6))
plt.subplot(121)
plot_acf(time_series, ax=plt.gca(), lags=40)

# Plot the Partial Autocorrelation Function (PACF)
plt.subplot(122)
plot_pacf(time_series, ax=plt.gca(), lags=40)

plt.savefig('acf_pacf.png')

