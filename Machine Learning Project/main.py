"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load your data
data = pd.read_csv('C:\\Users\\siddh\\Downloads\\Customer-Lifetime-Value-Prediction-in-python-main\\Customer-Lifetime-Value-Prediction-in-python-main\\online_retail_dataset.csv')

# Calculate Recency, Frequency, and Monetary Value (RFM)
current_date = datetime.today()
data['InvoiceDate'] = pd.to_datetime(data['InvoiceDate'])
recency = data.groupby('CustomerID')['InvoiceDate'].max()
data['Recency'] = (current_date - recency).dt.days
frequency = data.groupby('CustomerID')['InvoiceDate'].count()
data['Frequency'] = frequency
monetary_value = data.groupby('CustomerID')['UnitPrice'].sum()
data['Monetary_Value'] = monetary_value
data['Log_Monetary_Value'] = np.log1p(data['Monetary_Value'])

# Create scatter plots for RFM
plt.figure(figsize=(15, 5))

# Recency vs. Monetary Value
plt.subplot(131)
plt.scatter(data['Recency'], data['Log_Monetary_Value'], alpha=0.5)
plt.title('Recency vs. Log(Monetary Value)')
plt.xlabel('Recency (Days)')
plt.ylabel('Log(Monetary Value)')

# Frequency vs. Monetary Value
plt.subplot(132)
plt.scatter(data['Frequency'], data['Log_Monetary_Value'], alpha=0.5)
plt.title('Frequency vs. Log(Monetary Value)')
plt.xlabel('Frequency')
plt.ylabel('Log(Monetary Value)')

# Histogram of Monetary Value
plt.subplot(133)
plt.hist(data['Log_Monetary_Value'], bins=30, alpha=0.7)
plt.title('Histogram of Log(Monetary Value)')
plt.xlabel('Log(Monetary Value)')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()

# Prepare the features and target variable
X = data[['Recency', 'Frequency', 'Log_Monetary_Value']]
y = data['Monetary_Value']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")

# Plot actual vs. predicted values
plt.figure(figsize=(8, 5))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.title('Actual vs. Predicted CLV')
plt.xlabel('Actual CLV')
plt.ylabel('Predicted CLV')
plt.show()

# Example: Predict CLV for a specific customer
customer_data = [[90, 5, np.log1p(2000)]]
clv_prediction = model.predict(customer_data)
print(f"Predicted CLV: {clv_prediction[0]}")
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load data
data = pd.read_csv('C:\\Users\\siddh\\Downloads\\Customer-Lifetime-Value-Prediction-in-python-main\\Customer-Lifetime-Value-Prediction-in-python-main\\online_retail_dataset.csv', encoding='latin-1')

data['InvoiceDate'] = pd.to_datetime(data['InvoiceDate'], format='mixed', dayfirst=True, utc=True)

# FIX 1 & 3: Build a clean customer-level DataFrame using a timezone-safe reference date
current_date = pd.Timestamp('today', tz='UTC')

rfm = data.groupby('CustomerID').agg(
    Recency=('InvoiceDate', lambda x: (current_date - x.max()).days),
    Frequency=('InvoiceDate', 'count'),
    Monetary_Value=('UnitPrice', 'sum')
).reset_index()

rfm['Log_Monetary_Value'] = np.log1p(rfm['Monetary_Value'])

# Scatter plots for RFM
plt.figure(figsize=(15, 5))

plt.subplot(131)
plt.scatter(rfm['Recency'], rfm['Log_Monetary_Value'], alpha=0.5)
plt.title('Recency vs. Log(Monetary Value)')
plt.xlabel('Recency (Days)')
plt.ylabel('Log(Monetary Value)')

plt.subplot(132)
plt.scatter(rfm['Frequency'], rfm['Log_Monetary_Value'], alpha=0.5)
plt.title('Frequency vs. Log(Monetary Value)')
plt.xlabel('Frequency')
plt.ylabel('Log(Monetary Value)')

plt.subplot(133)
plt.hist(rfm['Log_Monetary_Value'], bins=30, alpha=0.7)
plt.title('Histogram of Log(Monetary Value)')
plt.xlabel('Log(Monetary Value)')
plt.ylabel('Count')

plt.tight_layout()
plt.show()

# FIX 2: Remove Log_Monetary_Value from features to prevent data leakage
X = rfm[['Recency', 'Frequency']]
y = rfm['Monetary_Value']

# Split, train, evaluate
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")

# Actual vs Predicted plot
plt.figure(figsize=(8, 5))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.title('Actual vs. Predicted CLV')
plt.xlabel('Actual CLV')
plt.ylabel('Predicted CLV')
plt.show()

# FIX 2 continued: Predict using only Recency and Frequency
customer_data = [[90, 5]]  # Recency=90 days, Frequency=5 purchases
clv_prediction = model.predict(customer_data)
print(f"Predicted CLV: {clv_prediction[0]:.2f}")