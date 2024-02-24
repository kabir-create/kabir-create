# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import joblib

# Load the dataset (you can replace this with your own dataset)
# For simplicity, let's assume a dataset with features like 'brand', 'model', 'year', 'mileage', 'condition', and 'price'
# In practice, you would have a more extensive dataset with additional features
data = pd.read_csv('car_dataset.csv')

# Preprocess the data (handle missing values, encode categorical variables, etc.)
# This is a simplified example, and you may need to perform more detailed preprocessing based on your dataset

# Assuming 'brand' and 'condition' are categorical variables, encode them
data = pd.get_dummies(data, columns=['brand', 'condition'], drop_first=True)

# Select features and target variable
features = ['brand_Toyota', 'brand_Ford', 'brand_Honda', 'condition_Good', 'condition_Excellent', 'year', 'mileage']
target = 'price'

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data[features], data[target], test_size=0.2, random_state=42)

# Train a Random Forest Regressor model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model (MSE in this example)
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Save the trained model for future use in the app
joblib.dump(model, 'car_price_prediction_model.joblib')
