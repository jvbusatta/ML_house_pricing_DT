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
melbourne_model = DecisionTreeRegressor(random_state=1)

print('------------------')
melbourne_model.fit(X,y)
print("Making predictions for the following 5 houses:")
print(X.head())
print("The predictions are")
print(melbourne_model.predict(X.head()))
