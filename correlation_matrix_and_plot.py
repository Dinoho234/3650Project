import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

df = pd.read_excel('data.xlsx', sheet_name='in',skiprows=11, nrows=278)

df.replace("N.A.", np.nan, inplace=True)

# Carbon Monoxide and Fine Suspended Particulates has the most null value, we drop the 2 columns  
# also drop station and year column for analysis
df.drop(['FSP', 'CO', 'STATION', 'YEAR'], axis=1, inplace=True)

# turning data into float type
df = df.astype('float')

#handling missing data filling mean value into null value columns
df.fillna(df.mean(), inplace=True)

metereological_data = ['Pressure', 'Temperature', 'Dew_Point', 'Humidity', 'Cloud', 'Rainfall', 'Visibility_Hours', 'Sunshine', 'Radiation','Evaporation', 'Wind_Direction', 'Wind_Speed']

# X contain air pollutans SO2, NOX, NO2, RSP, O3
X = df.drop(metereological_data, axis=1)

for i in range(len(metereological_data)):
    Y = df[metereological_data[i]]
    corr_matrix = X.corrwith(Y)

    print(metereological_data[i])
    print(corr_matrix)


# Set up a color palette for the scatter plots
colors = sns.color_palette('husl', len(X.columns))

# Create a figure and axis object
fig, ax = plt.subplots(figsize=(10, 8))

# Loop through each air pollutant in X
for i, col in enumerate(X.columns):
    # Calculate the correlation with Y
    corr = np.corrcoef(X[col], Y)[0, 1]
    
    # Fit a linear regression model to the data
    lr = LinearRegression()
    lr.fit(X[[col]], Y)
    
    # Get the slope and intercept of the regression line
    slope = lr.coef_[0]
    intercept = lr.intercept_
    
    # Plot a scatter plot of the two variables with a different color for each pollutant
    ax.scatter(X[col], Y, color=colors[i], label=f'{col} (corr = {corr:.2f})')
    
    # Plot the regression line
    ax.plot(X[col], slope*X[col] + intercept, color=colors[i], linestyle='--')

# Set the axis labels and legend
ax.set_title('Relationship between Air Pollutant and wind speed')
ax.set_xlabel('Air Pollutant')
ax.set_ylabel(metereological_data[-1])
ax.legend()

# Show the plot
plt.show()