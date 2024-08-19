from sklearn.datasets import fetch_california_housing
import pandas as pd
# Load the dataset as a data frame
california_housing = fetch_california_housing(as_frame=True)
df = california_housing.data
df.plot(kind="box", subplots=True, layout=(3,3), figsize=(10,10))

#Box plot

import plotly.express as px
fig = px.histogram(df, x="MedInc", marginal="box")
fig.show()

import seaborn as sns
iris = sns.load_dataset("iris")
ax = sns.boxplot(data=iris, palette="Set2")

#Scatter plots

import matplotlib.pyplot as plt
plt.scatter(df.AveRooms, df.AveBedrms, alpha=0.5)
plt.xlabel('AveRooms')
plt.ylabel('AveBedrms')
plt.title('AveBedrms - AveRooms')
plt.show()

import matplotlib.pyplot as plt
plt.scatter(df.HouseAge, df.Population, alpha=0.5)
plt.xlabel('HouseAge')
plt.ylabel('Population')
plt.title('Population - HouseAge')
plt.show()

#Interquartile Range IQR

# calculate Q1 and Q3
Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
# calculate the IQR
IQR = Q3 - Q1
# filter the dataset with the IQR
IQR_outliers = df[((df < (Q1-1.5*IQR)) |
(df > (Q3+1.5*IQR))).any(axis=1)]
IQR_outliers

df = df[~((df < (Q1-1.5*IQR)) |(df > (Q3+1.5* IQR))).any(axis=1)]


#Z-score

from scipy import stats
import numpy as np
# Calculate the z-scores
z_scores = stats.zscore(df)
z_scores

#Interquartile Range IQR

# Convert to absolute values
abs_z_scores = np.abs(z_scores)
# Select data points with a z-scores above or below 3
filtered_entries = (abs_z_scores < 3).all(axis=1)
# Filter the dataset
df_wo_outliers = df[filtered_entries]
df_wo_outliers.shape
(19794, 8)