import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from math import radians, cos, sin, asin, sqrt
from sklearn import model_selection
from sklearn import tree, ensemble, model_selection, metrics
from sklearn.model_selection import cross_val_score

# View the basic information of the dataset
taxi_train = pd.read_csv('taxi-train.csv')
taxi_text = pd.read_csv('taxi-test.csv')

print(taxi_train.shape)
print(taxi_text.shape)
print(taxi_train.head())

# Basic data processing
# vendor_id one-hot encoding
print(taxi_train['vendor_id'].value_counts())

label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(taxi_train['vendor_id'])
onehot_encoder = OneHotEncoder(sparse=False, )
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
onehot_encoded = pd.DataFrame(onehot_encoded)
taxi_train = pd.concat([taxi_train, onehot_encoded], axis=1)

print(taxi_train.columns)
# change the column name of one-hot encode for vendor_id
taxi_train.columns = ['vendor_id', 'pickup_datetime', 'dropoff_datetime', 'pickup_longitude', 'pickup_latitude',
                      'dropoff_longitude',
                      'dropoff_latitude', 'rate_code', 'passenger_count', 'trip_distance', 'payment_type',
                      'fare_amount', 'tip_amount',
                      'tip_paid', 'vendor_id_CMT', 'vendor_id_DDS', 'vendor_id_VTS']
del taxi_train['vendor_id']

# Convert the pickup time to hours, and group the time periods into one-hot encoding
taxi_train.pickup_datetime
taxi_train['pickup_hour'] = taxi_train['pickup_datetime'].astype("str").str[11:13]
taxi_train['pickup_hour'] = taxi_train['pickup_hour'].astype('int')


def cal_hour(x):
    if x - 4 < 0:
        return x - 4 + 24
    else:
        return x - 4


taxi_train.pickup_hour = taxi_train.pickup_hour.apply(cal_hour)


def Bucketize(x):
    if 0 <= x <= 7:
        return "morning"
    elif 7 < x <= 12:
        return "noon"
    elif 12 < x <= 18:
        return "afternoon"
    else:
        return "evening"


taxi_train.pickup_hour = taxi_train.pickup_hour.apply(Bucketize)

integer_encoded = label_encoder.fit_transform(taxi_train['pickup_hour'])
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
onehot_encoded = pd.DataFrame(onehot_encoded)
taxi_train = pd.concat([taxi_train, onehot_encoded], axis=1)

print(taxi_train.columns)
taxi_train.columns = ['pickup_datetime', 'dropoff_datetime', 'pickup_longitude', 'pickup_latitude', 'dropoff_longitude',
                      'dropoff_latitude', 'rate_code', 'passenger_count', 'trip_distance', 'payment_type',
                      'fare_amount', 'tip_amount',
                      'tip_paid', 'vendor_id_CMT', 'vendor_id_DDS', 'vendor_id_VTS', 'pickup_hour', 'afternoon',
                      'evening',
                      'morning', 'noon']
del taxi_train['pickup_hour']

# Calculate the time difference (in seconds)
taxi_train['pickup_datetime'] = pd.to_datetime(taxi_train['pickup_datetime'])
taxi_train['dropoff_datetime'] = pd.to_datetime(taxi_train['dropoff_datetime'])
taxi_train['duration'] = (taxi_train['dropoff_datetime'] - taxi_train['pickup_datetime']).dt.seconds

del taxi_train['pickup_datetime']
del taxi_train['dropoff_datetime']
del taxi_train['rate_code']
taxi_train


# Calculate the distance between getting on and off the bus from the city center, the latitude and
# longitude of New York city center is (-74.0059731, 40.7143528)


def distance(lon1, lat1):
    lon2 = -74.0059731
    lat2 = 40.7143528
    lon1, lat1, lon2, lat2 = map(radians, [float(lon1), float(lat1), float(lon2), float(lat2)])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    r = 6371.137
    return float('%.2f' % (c * r))


taxi_train['central_pick'] = taxi_train.apply(lambda row: distance(row['pickup_longitude'], row['pickup_latitude']),
                                              axis=1)
taxi_train['central_drop'] = taxi_train.apply(lambda row: distance(row['dropoff_longitude'], row['dropoff_latitude']),
                                              axis=1)
taxi_train.drop(taxi_train.columns[[0, 1, 2, 3]], axis=1, inplace=True)

# payment_type one hot encoding
taxi_train.payment_type = taxi_train.payment_type.fillna('Dis')


def buck(x):
    if x == 'CSH' or x == 'CAS' or x == 'Cas':
        return 'CSH'
    elif x == 'CRD' or x == 'CRE' or x == 'Cre':
        return 'CRD'
    else:
        return 'UNK'


taxi_train.payment_type = taxi_train.payment_type.apply(buck)
integer_encoded = label_encoder.fit_transform(taxi_train['payment_type'])
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
onehot_encoded = pd.DataFrame(onehot_encoded)
taxi_train = pd.concat([taxi_train, onehot_encoded], axis=1)
taxi_train.columns = ['passenger_count', 'trip_distance', 'payment_type', 'fare_amount', 'tip_amount', 'tip_paid',
                      'vendor_id_CMT', 'vendor_id_DDS',
                      'vendor_id_VTS', 'afternoon', 'evening', 'morning', 'noon', 'duration', 'central_pick',
                      'central_drop', 'CRD', 'CSH', 'UNK']
del taxi_train['payment_type']
del taxi_train['tip_paid']

# Dataset normalization processing
taxi_train = (taxi_train - taxi_train.min()) / (taxi_train.max() - taxi_train.min())
taxi_train = pd.DataFrame(taxi_train)

# Split the taxi-train.csv into train, valid and test dataset.
# Divide the original data into training set and test set
taxi_train['passenger_amount'] = taxi_train['passenger_count']
taxi_train['distance'] = taxi_train['trip_distance']
del taxi_train['passenger_count']
del taxi_train['trip_distance']
x = taxi_train.iloc[:, 2:]
y_TipAmount = taxi_train.iloc[:, 1]
y_FareAmount = taxi_train.iloc[:, 0]

xTip_train, xTip_test, yTip_train, yTip_test = model_selection.train_test_split(x, y_TipAmount, test_size=0.2,
                                                                                random_state=2020)
xFare_train, xFare_test, yFare_train, yFare_test = model_selection.train_test_split(x, y_FareAmount, test_size=0.2,
                                                                                    random_state=2020)
# Model(for tip)
# RandomForestRegressor
# Parameter tuning
# Default parameter regression accuracy
rf0 = ensemble.RandomForestRegressor(oob_score=True, random_state=2020)
rf0.fit(xTip_train, yTip_train)
print("accuracy:%f" % rf0.score(xTip_train, yTip_train))

param_test1 = {"n_estimators": [10, 20, 30]}
gsearch1 = model_selection.GridSearchCV(estimator=ensemble.RandomForestRegressor(), param_grid=param_test1,
                                        scoring='neg_mean_squared_error', cv=5)
gsearch1.fit(xTip_train, yTip_train)

print(gsearch1.best_score_)
print(gsearch1.best_params_)
print("best accuracy:%f" % gsearch1.best_score_)

param_test2 = {"max_features": range(1, 11, 2)}
gsearch2 = model_selection.GridSearchCV(estimator=ensemble.RandomForestRegressor(n_estimators=30, random_state=2020),
                                        param_grid=param_test2,
                                        scoring='neg_mean_squared_error', cv=5)
gsearch2.fit(xTip_train, yTip_train)
print(gsearch2.best_score_)
print(gsearch2.best_params_)
print('best accuracy:%f' % gsearch2.best_score_)

rf = ensemble.RandomForestRegressor(n_estimators=30, max_features=3, oob_score=True, random_state=2020)
rf.fit(xTip_train, yTip_train)
print("accuracy: %f" % rf.score(xTip_train, yTip_train))

# Feature selection
importances = list(rf.feature_importances_)
feature_list = xTip_train.columns
feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]
print(type(feature_importances))
feature_importances = sorted(feature_importances, key=lambda x: x[1], reverse=True)
print(feature_importances)

# Testing Score
print("Testing Score:%f" % rf.score(xTip_test, yTip_test))

# AdaBoostRegressor
# Parameter tuning
# Default parameter regression accuracy
ab0 = ensemble.AdaBoostRegressor(random_state=2020)
ab0.fit(xTip_train, yTip_train)
print("accuracy:%f" % ab0.score(xTip_train, yTip_train))

n_estimators = [10, 20, 30]
parameters = dict(n_estimators=n_estimators)
gsearch1 = model_selection.GridSearchCV(estimator=ensemble.AdaBoostRegressor(), param_grid=parameters,
                                        scoring='neg_mean_squared_error', cv=5)
gsearch1.fit(xTip_train, yTip_train)

print(gsearch1.best_score_)
print(gsearch1.best_params_)
print("best accuracy:%f" % gsearch1.best_score_)

ab = ensemble.AdaBoostRegressor(n_estimators=10, random_state=2020)
ab.fit(xTip_train, yTip_train)
print("accuracy: %f" % ab.score(xTip_train, yTip_train))

# Feature selection
importances = list(ab.feature_importances_)
feature_list = xTip_train.columns
feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]
print(type(feature_importances))
feature_importances = sorted(feature_importances, key=lambda x: x[1], reverse=True)
print(feature_importances)

# Testing Score
print("Testing Score:%f" % ab.score(xTip_test, yTip_test))

# GradientBoostingRegressor
#  Parameter tuning
# Default parameter regression accuracy
gboost = ensemble.GradientBoostingRegressor(random_state=2019)
gboost.fit(xTip_train, yTip_train)
print("accuracy:%f" % gboost.score(xTip_train, yTip_train))

n_estimators = [10, 20, 30]
learning_rate = [0.01, 0.1]
parameters = dict(n_estimators=n_estimators, learning_rate=learning_rate)

gboost = model_selection.GridSearchCV(estimator=ensemble.GradientBoostingRegressor(), param_grid=parameters,
                                      scoring='neg_mean_squared_error', cv=5)
gboost.fit(xTip_train, yTip_train)

print(gboost.best_score_)
print(gboost.best_params_)
print("best accuracy:%f" % gboost.best_score_)

GB = ensemble.GradientBoostingRegressor(n_estimators=30, learning_rate=0.1, random_state=2020)
GB.fit(xTip_train, yTip_train)
print("accuracy: %f" % GB.score(xTip_train, yTip_train))

# Feature selection
importances = list(GB.feature_importances_)
feature_list = xTip_train.columns
feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]
print(type(feature_importances))
feature_importances = sorted(feature_importances, key=lambda x: x[1], reverse=True)
print(feature_importances)

# Testing Score
print("Testing Score:%f" % GB.score(xTip_test, yTip_test))
