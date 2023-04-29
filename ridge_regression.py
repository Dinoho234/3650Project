import pandas as pd
import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
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

## Y contain air pollutans SO2, NOX, NO2, RSP, O3
x = df['SO2'].values.reshape(-1, 1)
y = df['Visibility_Hours']

for i in range(1,9):
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=i/10)

    # Scale data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Fit Ridge Regression model on training set with cross validation
    ridge = RidgeCV(alphas=[ 0.1, 1, 10, 100])
    ridge.fit(X_train, y_train)

    # Predict on test set
    y_pred = ridge.predict(X_test)

    # Evaluate model performance
    mse = mean_squared_error(y_test, y_pred)
    print('MSE', 'for test size = ',i/10, ' ',mse)