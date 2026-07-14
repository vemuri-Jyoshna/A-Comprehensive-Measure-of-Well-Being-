import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle

# 1. Reading the Dataset
df = pd.read_csv("dataset/human_development_index.csv")
print("Dataset Shape:", df.shape)
print(df.head())

# Take the first 20 rows for clean, non-overcrowded visualization
data1 = df.head(20)

# 2. Data Visualization
# Verify unique countries
print("Unique Countries in subset:", data1['Country'].unique())

# Mean Years of Schooling vs HDI Score Strip Plot
plt.figure(figsize=(10, 6))
sns.stripplot(x='Mean years of schooling', y='Human Development Index (HDI) ', data=data1) # Adjust column names based on your CSV
plt.title('Mean Years of Schooling vs HDI Score')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Life Expectancy vs HDI
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Life expectancy at birth', y='Human Development Index (HDI) ', data=data1)
plt.title('Life Expectancy vs HDI Score')
plt.tight_layout()
plt.show()

# 3. Selecting Dependent and Independent Variables
# Assume column indices based on standard HDI datasets: 
# Target (Y) = HDI Score. Inputs (X) = Life Expectancy, Schooling, GNI per capita, etc.
# Modify the column names below to exactly match your downloaded CSV file headers
X = df[['Life expectancy at birth', 'Expected years of schooling', 'Mean years of schooling', 'Gross national income (GNI) per capita']]
Y = df['Human Development Index (HDI) ']

# 4. Checking and Handling Null Values
print("Missing values before mitigation:\n", X.isnull().sum())
X = X.fillna(X.mean())
print("Missing values after mitigation:\n", X.isnull().sum())

# 5. Dividing the Dataset into Train and Test Data
#  CORRECTED LINE:
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# 6. Fitting the Linear Regression Model
model = LinearRegression()
model.fit(X_train, Y_train)

# 7. Predicting and Validating Results
Y_pred = model.predict(X_test)
print("Sample Predictions:\n", Y_pred[:5])
print("Actual Target Values:\n", Y_test.values[:5])
print("Model R-Squared Score:", model.score(X_test, Y_test))

# 8. Saving the Model using Serialization (Pickle)
os.makedirs('models', exist_ok=True)
with open('models/hdi_model.pkl', 'wb') as file:
    pickle.dump(model, file)

print("Model successfully trained and saved into models/hdi_model.pkl")