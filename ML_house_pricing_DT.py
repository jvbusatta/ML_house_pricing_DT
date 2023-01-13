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

print('MAE(DECISION TREE REGRESSOR wout split): ',mean_absolute_error(y, predicted_home_prices))
print(' ')
#model validation

from sklearn.model_selection import train_test_split

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0)

#Define model
melbourne_model = DecisionTreeRegressor()

#Fit model
melbourne_model.fit(train_X, train_y)
val_predictions = melbourne_model.predict(val_X)
print('MAE(DECISION TREE REGRESSOR): ', mean_absolute_error(val_y, val_predictions))
print(' ')
#controlling the tree deep
from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor

def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes,random_state=0)
    model.fit(train_X, train_y)
    preds_val = model.predict(val_X)
    mae = mean_absolute_error(val_y, preds_val)
    return mae

#comparing different mae valeus according to values of max_leaf_nodes:
for max_leaf_nodes in [5, 50, 500, 5000]:
    my_mae = get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y)
    print("Max leaf nodes: %d  \t\t Mean Absolute Error:  %d" %(max_leaf_nodes, my_mae))

#here we can notice that 500 is the most acceptable value of leaves

#now using random forest to evaluate the performance of prediction
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

forest_model = RandomForestRegressor(random_state=1)
forest_model.fit(train_X, train_y)
melb_preds = forest_model.predict(val_X)
print('MAE(RANDOM FOREST): ', mean_absolute_error(val_y, melb_preds))
#We can see with this information that Random Forest is a better option since it's MAE value is smaller than using the other method 

#cleaning missing values
# Load the data
from sklearn.impute import SimpleImputer

#define the function to compare values
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# Function for comparing different approaches
def score_dataset(X_train, X_valid, y_train, y_valid):
    model = RandomForestRegressor(n_estimators=10, random_state=0)
    model.fit(X_train, y_train)
    preds = model.predict(X_valid)
    return mean_absolute_error(y_valid, preds)

data = pd.read_csv('D:\GitHub\ML_house_pricing_DT\melb_data.csv')

# Select target
y = data.Price

# To keep things simple, we'll use only numerical predictors
melb_predictors = data.drop(['Price'], axis=1)
X = melb_predictors.select_dtypes(exclude=['object'])

# Divide data into training and validation subsets
X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2,
                                                      random_state=0)

# Get names of columns with missing values
cols_with_missing = [col for col in X_train.columns
                     if X_train[col].isnull().any()]

# Drop columns in training and validation data
reduced_X_train = X_train.drop(cols_with_missing, axis=1)
reduced_X_valid = X_valid.drop(cols_with_missing, axis=1)

print("MAE from Approach 1 (Drop columns with missing values):")
print(score_dataset(reduced_X_train, reduced_X_valid, y_train, y_valid))

#second method (imputation)

from sklearn.impute import SimpleImputer

# Imputation
my_imputer = SimpleImputer()
imputed_X_train = pd.DataFrame(my_imputer.fit_transform(X_train))
imputed_X_valid = pd.DataFrame(my_imputer.transform(X_valid))

# Imputation removed column names; put them back
imputed_X_train.columns = X_train.columns
imputed_X_valid.columns = X_valid.columns

print("MAE from Approach 2 (Imputation):")
print(score_dataset(imputed_X_train, imputed_X_valid, y_train, y_valid))

#third method(An Extension to Imputation)

# Make copy to avoid changing original data (when imputing)
X_train_plus = X_train.copy()
X_valid_plus = X_valid.copy()

# Make new columns indicating what will be imputed
for col in cols_with_missing:
    X_train_plus[col + '_was_missing'] = X_train_plus[col].isnull()
    X_valid_plus[col + '_was_missing'] = X_valid_plus[col].isnull()

# Imputation
my_imputer = SimpleImputer()
imputed_X_train_plus = pd.DataFrame(my_imputer.fit_transform(X_train_plus))
imputed_X_valid_plus = pd.DataFrame(my_imputer.transform(X_valid_plus))

# Imputation removed column names; put them back
imputed_X_train_plus.columns = X_train_plus.columns
imputed_X_valid_plus.columns = X_valid_plus.columns

print("MAE from Approach 3 (An Extension to Imputation):")
print(score_dataset(imputed_X_train_plus, imputed_X_valid_plus, y_train, y_valid))

# Shape of training data (num_rows, num_columns)
print(X_train.shape)

# Number of missing values in each column of training data
missing_val_count_by_column = (X_train.isnull().sum())
print(missing_val_count_by_column[missing_val_count_by_column > 0])


