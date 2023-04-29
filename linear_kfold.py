import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load data
df = pd.read_excel('data.xlsx', sheet_name='in',skiprows=11, nrows=278)

df.replace("N.A.", np.nan, inplace=True)

# Carbon Monoxide and Fine Suspended Particulates have the most null values, so drop those 2 columns
# also drop station and year column for analysis
df.drop(['FSP', 'CO', 'STATION', 'YEAR'], axis=1, inplace=True)

# turning data into float type
df = df.astype('float')

#handling missing data filling mean value into null value columns
df.fillna(df.mean(), inplace=True)

## x contains concentration of air pollutants SO2
x = df['SO2'].values.reshape(-1, 1)
y = df['Visibility_Hours']

# Create a Linear Regression object
model = LinearRegression()

for k in range(2, 11):
    # Use k-fold cross validation
    kf = KFold(n_splits=k, shuffle=True, random_state=42)

    mse_values = []

    for train_index, test_index in kf.split(x):
        X_train, X_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Train the model on the training data
        model.fit(X_train, y_train)

        # Make predictions on the test data
        y_pred = model.predict(X_test)

        # Evaluate the performance of the model using MSE
        mse = mean_squared_error(y_test, y_pred)
        mse_values.append(mse)

    # Calculate the average MSE across all 10 folds
    average_mse = np.mean(mse_values)
    print('Average MSE:','for k = ', i,' ', average_mse)
