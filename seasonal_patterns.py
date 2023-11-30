from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt
import pandas as pd


data = pd.read_csv('data_daily.csv', parse_dates=['# Date'], index_col='# Date')

# Decompose the time series to observe any seasonal patterns
decomposition = seasonal_decompose(data['Receipt_Count'], model='additive', period=1)

# Plot the decomposed components of the time series
fig = decomposition.plot()
fig.set_size_inches(14, 7)
plt.show()


data_new = pd.read_csv('data_daily.csv')

# Convert the 'Date' column to datetime and sort by date
data_new['# Date'] = pd.to_datetime(data_new['# Date'])
data_new.sort_values('# Date', inplace=True)

# Calculate the differences to find potential spikes (large increases or decreases)
data_new['Diff'] = data_new['Receipt_Count'].diff().abs()

# We will consider significant spikes where the difference is above a certain percentile
threshold_percentile = 95
threshold_value = data_new['Diff'].quantile(threshold_percentile / 100.0)

# Identifying the dates of these spikes
spikes = data_new[data_new['Diff'] > threshold_value]

# Calculate the intervals between the spikes in terms of days
spikes['Interval'] = spikes['# Date'].diff().dt.days

# Look at the intervals to find the most common pattern
common_intervals = spikes['Interval'].value_counts()
common_intervals.head()
print(common_intervals.head())
