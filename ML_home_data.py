import pandas as pd
iowa_file_path = 'D:/GitHub/ML_house_pricing_DT/train.csv'

home_data = pd.read_csv(iowa_file_path)

print(home_data)

feature_names = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
X = feature_names

print(X)