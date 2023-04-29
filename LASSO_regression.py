import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error

# Load data
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

## Y contain air pollutans SO2, NOX, NO2, RSP, O3
x = df.drop(metereological_data, axis=1)
y = df['Visibility_Hours']
# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# Create a Linear Regression object
model = Lasso()

# Train the model on the training data
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Evaluate the performance of the model using MSE
mse = mean_squared_error(y_test, y_pred)
print('MSE:', mse)