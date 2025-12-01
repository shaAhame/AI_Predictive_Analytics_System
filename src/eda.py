import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import missingno as ms
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
import pickle

df=pd.read_csv("SuperstoresSales.csv",encoding="ISO-8859-1")
df.head()

print(df.head(5))

print(df.info())

print(df.columns)

print(df.shape)

print(df.duplicated().sum())

print(df.isnull().sum())



ms.bar(df,figsize = (10,5),color="tomato")
plt.title("Bar plot showing missing data values", size = 15,c="r")
plt.show()

print(df["Country"].value_counts())
print(df["Segment"].value_counts())

sns.countplot(x="Segment", data=df)
plt.title("count of customers by segment")
plt.show()

print(df["City"].value_counts())


sns.countplot(x="City", data=df)
plt.title("count of customers by city")
plt.show()

print(df["Region"].value_counts())

sns.countplot(x="Region", data=df)
plt.title("count of customers by region")
plt.show()

print(df["Category"].value_counts())

sns.countplot(x="Category", data=df)
plt.title("count of customers by category")
plt.show()

# Convert timestamp to datetime
df['Order Date'] = pd.to_datetime(df['Order Date'])




# Extract time-based features
df['Hour'] = df['Order Date'].dt.hour
df['Day'] = df['Order Date'].dt.day
df['Month'] = df['Order Date'].dt.month
df['Season'] = df['Month'].apply(lambda x: 'Winter' if x in [12, 1, 2] else
                                 'Spring' if x in [3, 4, 5] else
                                 'Summer' if x in [6, 7, 8] else 'Autumn')



# Plot Sales by season

plt.figure(figsize=(10, 6))
sns.boxplot(x='Season', y='Sales', data=df)
plt.title('Seasonal Variation in Sales')
plt.xlabel('Season')
plt.ylabel('Sales')
plt.show()


# Plot Profit by season
plt.figure(figsize=(10, 6))
sns.boxplot(x='Season', y='Profit', data=df)
plt.title('Seasonal Variation in Profit')
plt.xlabel('Season')
plt.ylabel('Profit')
plt.show()

# Plot the sales montly
plt.figure(figsize=(12, 6))
sns.lineplot(x='Month', y='Sales', data=df, ci=None)
plt.title('Monthly Variation in Sales')
plt.xlabel('Month')
plt.ylabel('Sales')
plt.show()


#Time Series Decomposition of Sales Data
# Set the frequency (e.g., daily, hourly)


decomposition = seasonal_decompose(df['Sales'], model='additive', period=24)  # Daily seasonality

# Plot decomposition
plt.figure(figsize=(12, 8))
decomposition.plot()
plt.suptitle('Time Series Decomposition of Sales Data')
plt.show()

