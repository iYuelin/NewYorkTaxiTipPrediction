# NewYorkTaxiTipPrediction
New York Taxi Tip Prediction：
Build ensemble models to predict two things a taxi trip in New York City, based on other trip information, such as geo location, time of the trip, etc.

In this machine learning task, interpret recall as r2_score.

Random Forest
Objective metric on test: 0.662110
Parameters: {'n_estimators': 30}{'max_features': 3}

AdaBoost
Objective metric on test: 0.557304
Parameters: {'n_estimators': 30}

Gradient Boosting
Objective metric on test: 0.678982
Parameters: {'learning_rate': 0.1, 'n_estimators': 30}

Important Features：
Classifier	RandomForest	AdaBoost	   GradientBoosting
Feature 1	'CRD'	        'CRD'	           'CRD'
Feature 2	'CSH'	        'central_pick'	   'duration'
Feature 3	'duration'	'duration'	   'distance'
Feature 4	'distance'	'passenger_amount' 'vendor_id_CMT'
Feature 5	'central_pick'	'distance'	   'vendor_id_DDS'
