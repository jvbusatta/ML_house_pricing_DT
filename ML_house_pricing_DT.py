import pandas as pd

melbourne_file_path = 'D:\GitHub\ML_house_pricing_DT\melb_data.csv'

melbourne_data = pd.read_csv(melbourne_file_path)

print(melbourne_data.describe())

print(melbourne_data.columns)

melbourne_data = melbourne_data.dropna(axis=0)

#target
y = melbourne_data.Price
print(y)

melbourne_features = ['Rooms','Bathroom', 'Landsize', 'Lattitude', 'Longtitude']

X = melbourne_data[melbourne_features]

print(X.describe())

print(X.head())

#next step to choose the ML model
#Define
#Fit
#Predict
#Evaluate

from sklearn.tree import DecisionTreeRegressor
#defining model
melbourne_model = DecisionTreeRegressor(random_state=1)

print('------------------')
#fitting model
melbourne_model.fit(X,y)
print("Making predictions for the following 5 houses:")
print(X.head())

print("The predictions are")
print(melbourne_model.predict(X.head()))

#calculating the Mean Absulute Error (MAE)
from sklearn.metrics import mean_absolute_error

predicted_home_prices = melbourne_model.predict(X)

print(mean_absolute_error(y, predicted_home_prices))

#model validation

from sklearn.model_selection import train_test_split

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0)

#Define model
melbourne_model = DecisionTreeRegressor()

#Fit model
melbourne_model.fit(train_X, train_y)
val_predictions = melbourne_model.predict(val_X)
print(mean_absolute_error(val_y, val_predictions))

    
