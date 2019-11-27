'''import data using pandas library '''
import numpy as np
import pandas as pd

''' import data visualization libraries '''
import matplotlib.pyplot as plt
import seaborn as sns
#%matplotlib inline
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.linear_model import lasso_path, enet_path
from sklearn.linear_model import Lasso
from sklearn.metrics import r2_score
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import SGDClassifier
from sklearn import svm
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KNeighborsClassifier

''' open the data file and present first 5 rows '''
df = pd.read_csv('Z:\COMP 3710_01 - Applied Artificial Intelligence (Fall 19 Park)\Project AI Crime\crime-in-vancouver\crime.csv')
# print(df.head())

''' drop 'MINUTE' columns since it will not be used in data representation '''

df_drop = df.drop(['MINUTE'], axis = 1, inplace = True)
#print(df.head()) 

''' check for data completeness '''
# df.info()

'''
 'RangeIndex: 530652 entries, 0 to 530651'
 'HOUR 476290 non-null float64'
 'NEIGHBOURHOOD 474028 non-null object'
 'HUNDRED_BLOCK 530639 non-null object'
  tells us that the data is incomplete in these areas.

  Need to fill in data identifies as 'Not given'
'''
df['HOUR'].fillna(-1.0, inplace = True)
df['NEIGHBOURHOOD'].fillna('Not Given', inplace = True)
df['HUNDRED_BLOCK'].fillna('Not Given', inplace = True)
df['HOUR'].replace(0.0, 24.0, inplace=True)
#print(df.tail(20))
'''
 since data is allocated in separated columns,
 need to put all data in one column named 'FULL_DATE'
'''
df['FULL_DATE'] = pd.to_datetime({'year': df['YEAR'], 'month': df['MONTH'],
'day':df['DAY']})

'''  use pandas.day_name() to determine the day of a crime / use .dayofweek for
 (Monday = 0, Tuesday = 1, etc.) '''
df['DAY_OF_WEEK'] = df['FULL_DATE'].dt.weekday_name

'''
using the date as indexes
'''
# df.index = pd.DatetimeIndex(df['FULL_DATE'])

''' 
filtering the data excluding the last days of the last month
the data set was extracted
'''
df = df[df['FULL_DATE'] < '2017-07-01']


''' categorising the types of crimes '''
# print('Categories of Crime and thier number of occurrences')
# print(df['TYPE'].value_counts().sort_index())

''' a function to sort out crime types on most occuring ones and others '''
def type(crime_type):
    if 'Theft' in crime_type:
        return 'Theft'
    elif 'Break' in crime_type:
        return 'Break and Enter'
    elif 'Homicide' in crime_type:
        return 'Homicide'
    elif 'Collision' in crime_type:
        return 'Vehicle Collision'
    else:
        return 'Others'

''' add a column named 'CATEGORY' to display the type of crime comminted '''
df['CATEGORY'] = df['TYPE'].apply(type)


'''
 associate 'Vehicle Collision' with an acident, not a crime 
 assign crimes to 'crime' var
 assign car collisions to 'car_coll' var
 '''
car_coll = df[df['CATEGORY'] == 'Vehicle Collision']
crime = df[df['CATEGORY'] != 'Vehicle Collision']

''' 
 creating a time series graph with crimes per day 
 excluding the outlier
'''
crime_per_day = pd.DataFrame(crime['FULL_DATE'] != '2011-06-11')


# ! -----------------------------------------------------------------------
''' Predicting a crime based on the TYPE, HUNDRED_BLOCKHUNDRED_BLOCK, NEIGHBOURHOOD,
 Latitude, Longitude, FULL_DATE attributes '''

''' Data Cleaning '''
# new_crime = crime.drop(['CATEGORY', 'DAY_OF_WEEK', 'YEAR', 'MONTH', 'DAY', 'HOUR', 'X', 'Y'], axis=1)

''' Data Preprocessing '''
# le = preprocessing.LabelEncoder()
# new_crime['TYPE'] = le.fit_transform(new_crime['TYPE'])
# new_crime['HUNDRED_BLOCK'] = le.fit_transform(new_crime['HUNDRED_BLOCK'])
# new_crime['NEIGHBOURHOOD'] = le.fit_transform(new_crime['NEIGHBOURHOOD'])
# new_crime['Latitude'] = le.fit_transform(new_crime['Latitude'])
# new_crime['Longitude'] = le.fit_transform(new_crime['Longitude'])
# new_crime['FULL_DATE'] = le.fit_transform(new_crime['FULL_DATE'])


''' Preparing data and target vars '''
# cols = [col for col in new_crime.columns if col not in ['TYPE']]
# data = new_crime[cols]
# target = new_crime['TYPE']

''' Splitting the data into training and testing sets '''
# data_train, data_test, target_train, target_test = train_test_split(data, target, test_size = 0.33, random_state = 42)

# print("data_train: ", data_train.shape)
# print("data_test: ", data_test.shape)
# print("target_train: ", target_train.shape)
# print("target_test: ", target_test.shape)

''' Naive-Bayes model for prediction'''
# gnb = GaussianNB()
# pred = gnb.fit(data_train, target_train).predict(data_test)
# print("Naive-Bayes accuracy : ",accuracy_score(target_test, pred, normalize = True))


'''  Lasso Model '''
#data /= data.std(axis=0)
#eps = 5e-3
#alphas_lasso, coefs_lasso, _ = lasso_path(data, target, eps, fit_intercept=False)

# alpha = 0.1
# lasso = Lasso(alpha=alpha)

# y_pred_lasso = lasso.fit(data_train, target_train).predict(data_test)
# r2_score_lasso = r2_score(target_test, y_pred_lasso)
# print(lasso)
# print("r^2 on test data : %f" % r2_score_lasso)

''' Elastic Net '''
#alphas_enet, coefs_enet, _ = enet_path(data, target, eps=eps, l1_ratio=0.8, fit_intercept=False)
# enet = ElasticNet(alpha=alpha, l1_ratio=0.7)

# y_pred_enet = enet.fit(data_train, target_train).predict(data_test)
# r2_score_enet = r2_score(target_test, y_pred_enet)
# print(enet)
# print("r^2 on test data : %f" % r2_score_enet)

# ! ---------------------------------------------------------------------------
''' Predicitng location of a crime via only Latitude, Longitude, X, Y params '''

''' Data cleaning '''
# loc_pred_df = crime.drop(['CATEGORY', 'DAY_OF_WEEK', 'YEAR', 'MONTH', 'DAY', 'HOUR', 'FULL_DATE', 'HUNDRED_BLOCK', 'NEIGHBOURHOOD' ], axis=1)

''' Data Preprocessing '''
# le = preprocessing.LabelEncoder()
# loc_pred_df['TYPE'] = le.fit_transform(loc_pred_df['TYPE'])
# loc_pred_df['Latitude'] = le.fit_transform(loc_pred_df['Latitude'])
# loc_pred_df['Longitude'] = le.fit_transform(loc_pred_df['Longitude'])
# loc_pred_df['X'] = le.fit_transform(loc_pred_df['X'])
# loc_pred_df['Y'] = le.fit_transform(loc_pred_df['Y'])

''' Preparing data and target vars '''
# loc_cols = [col for col in loc_pred_df.columns if col not in ['TYPE']]
# loc_data = loc_pred_df[loc_cols]
# loc_target = loc_pred_df['TYPE']

''' Splitting the data into training and testing sets '''
# data_train2, data_test2, target_train2, target_test2 = train_test_split(loc_data, loc_target, test_size = 0.33, random_state = 42)

# print("data_train: ", data_train2.shape)
# print("data_test: ", data_test2.shape)
# print("target_train: ", target_train2.shape)
# print("target_test: ", target_test2.shape)

'''  Lasso Model '''
# data /= data.std(axis=0)
# eps = 5e-3
# alphas_lasso, coefs_lasso, _ = lasso_path(loc_data, loc_target, eps, fit_intercept=False)

# alpha = 0.1
# lasso = Lasso(alpha=alpha)

# y_pred_lasso = lasso.fit(data_train2, target_train2).predict(data_test2)
# r2_score_lasso = r2_score(target_test2, y_pred_lasso)
# print(lasso)
# print("r^2 on test data : %f" % r2_score_lasso)


''' Stochalistic Gradient Descent Classification'''
# clf = SGDClassifier(loss="hinge", penalty="l2", shuffle=True, max_iter=10,)
# clf.fit(data_train2, target_train2) 
# print('SGD classification val: ', clf.predict(data_test2))

''' Stochalistic Gradient Descent Prediction'''
# clf2 = SGDClassifier(loss="log", penalty="l2", shuffle=True, max_iter=10,).fit(data_train2, target_train2) 
# print('SGD prediction val: ', clf2.predict_proba(data_test2))


# !--------------------------------------------------------------------
''' Prediction Date of a Crime with Month and Day attrivutes'''
# time_pred_df = crime.drop(['CATEGORY', 'DAY_OF_WEEK', 'FULL_DATE', 'HOUR', 'YEAR', 'HUNDRED_BLOCK', 'NEIGHBOURHOOD', 'X', 'Y', 'Latitude',
# 'Longitude'], axis=1)

''' Data Preprocessing '''
# le = preprocessing.LabelEncoder()
# time_pred_df['TYPE'] = le.fit_transform(time_pred_df['TYPE'])
# time_pred_df['MONTH'] = le.fit_transform(time_pred_df['MONTH'])
# time_pred_df['DAY'] = le.fit_transform(time_pred_df['DAY'])

''' Preparing data and target vars '''
# time_cols = [col for col in time_pred_df.columns if col not in ['TYPE']]
# time_data = time_pred_df[time_cols]
# time_target = time_pred_df['TYPE']

''' Splitting the data into training and testing sets '''
# data_train3, data_test3, target_train3, target_test3 = train_test_split(time_data, time_target, test_size = 0.33, random_state = 42)

# print("data_train: ", data_train3.shape)
# print("data_test: ", data_test3.shape)
# print("target_train: ", target_train3.shape)
# print("target_test: ", target_test3.shape)

'''  Lasso Model '''
# data /= data.std(axis=0)
# eps = 5e-3
# alphas_lasso, coefs_lasso, _ = lasso_path(time_data, time_target, eps, fit_intercept=False)

# alpha = 0.1
# lasso = Lasso(alpha=alpha)

# y_pred_lasso = lasso.fit(data_train3, target_train3).predict(data_test3)
# r2_score_lasso = r2_score(target_test3, y_pred_lasso)
# print(lasso)
# print("Lasso Model score on test data (time) : %f" % r2_score_lasso)


''' Stochalistic Gradient Descent Classification'''
# clf = SGDClassifier(loss="hinge", penalty="l2", shuffle=True, max_iter=10,)
# clf.fit(data_train3, target_train3) 
# print('SGD classification val: ', clf.predict(data_test3))

''' Stochalistic Gradient Descent Prediction'''
# clf2 = SGDClassifier(loss="log", penalty="l2", shuffle=True, max_iter=10,).fit(data_train3, target_train3) 
# print('SGD prediction val: ', clf2.predict_proba(data_test3))

# ! --------------------------------------------------------------------
# TODO: define a var as a crime type (take break in residential)
# TODO: define hour range for day and night

''' exclude the rows where 'HOUR' value is not given (i.e. replaced to -1.0) '''
df  = df[df.HOUR != -1.0]

''' define day and night ranges '''
def time_per(x):
    if (x >= 5.0) and (x < 18.0):
        return 'day'
    else:
        return 'night'

df['hour'] = df['HOUR'].apply(time_per)

''' Data Preprocessing '''
# clock = df.drop(['CATEGORY', 'DAY_OF_WEEK', 'FULL_DATE', 'DAY', 'HOUR', 'YEAR', 'HUNDRED_BLOCK', 'NEIGHBOURHOOD', 'X', 'Y', 'Latitude',
# 'Longitude'], axis=1)

''' define a crime type var '''
# break_res = clock.loc[df['TYPE'] == 'Break and Enter Residential/Other']

# lel = preprocessing.LabelEncoder()
# break_res['TYPE'] = lel.fit_transform(break_res['TYPE'])
# break_res['hour'] = lel.fit_transform(break_res['hour'])
# break_res['MONTH'] = lel.fit_transform(break_res['MONTH'])

''' Preparing data and target vars '''
# cols = [col for col in break_res.columns if col not in ['hour']]
# data = break_res[cols]
# target = break_res['hour']

''' Splitting the data into training and testing sets '''
# data_train4, data_test4, target_train4, target_test4 = train_test_split(data, target, test_size = 0.33, random_state = 42)

# print("data_train: ", data_train4.shape)
# print("data_test: ", data_test4.shape)
# print("target_train: ", target_train4.shape)
# print("target_test: ", target_test4.shape)

''' Stochalistic Gradient Descent Prediction'''
# clf3 = SGDClassifier(loss="log", penalty="l2", shuffle=True, max_iter=10,).fit(data_train4, target_train4) 
# print('SGD prediction val: ', clf3.predict_proba(data_test4))

''' SVM Classifier '''
# clf4 = svm.SVR()
# clf4.fit(data_train4, target_train4)
# print('SVM : ', clf4.predict(data_test4))


''' KNN '''
# knn = KNeighborsClassifier(n_neighbors=1)
# knn.fit(data_train4, target_train4)

# print('KNN val: ', knn.predict(data_test4))

# a = knn.predict(data_test4)
# unique, counts = np.unique(a, return_counts=True)

# print('{0 - Day(5am to 7pm), 1 - Night(6pm to 4am)}\n',dict(zip(unique, counts)))


# ! --------------------------------------------------------------------
''' 4 day-period check '''
def p(x):
    if (x >= 5.0) and (x <= 12.0):
        return 'Morning'
    elif (x > 12.0) and (x < 17.0):
        return 'Afternoon'
    elif (x >= 17.0) and (x < 21.0):
        return 'Evening'
    else:
        return 'Night'

df['DAY_PART'] = df['HOUR'].apply(p)

''' Data Preprocessing '''
part = df.drop(['CATEGORY', 'DAY_OF_WEEK', 'FULL_DATE', 'DAY', 'hour', 'HOUR', 'YEAR', 'HUNDRED_BLOCK', 'NEIGHBOURHOOD', 'X', 'Y', 'Latitude',
'Longitude'], axis=1)

''' define a crime type var '''
break_res2 = part.loc[df['TYPE'] == 'Break and Enter Residential/Other']

lel = preprocessing.LabelEncoder()
break_res2['TYPE'] = lel.fit_transform(break_res2['TYPE'])
break_res2['DAY_PART'] = lel.fit_transform(break_res2['DAY_PART'])
break_res2['MONTH'] = lel.fit_transform(break_res2['MONTH'])

''' Preparing data and target vars '''
cols = [col for col in break_res2.columns if col not in ['DAY_PART']]
data = break_res2[cols]
target = break_res2['DAY_PART']

''' Splitting the data into training and testing sets '''
data_train5, data_test5, target_train5, target_test5 = train_test_split(data, target, test_size = 0.33, random_state = 42)

print("data_train: ", data_train5.shape)
print("data_test: ", data_test5.shape)
print("target_train: ", target_train5.shape)
print("target_test: ", target_test5.shape)

''' KNN '''
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(data_train5, target_train5)
print('KNN val: ', knn.predict(data_test5))

a = knn.predict(data_test5)
unique, counts = np.unique(a, return_counts=True)

print('{0 - Afternoon(12pm to 5pm), 1 - Evening(5pm to 9pm), 2 - Morning(5am to 12pm(noon)), 3 - Night(9pm to 4 am)}\n', dict(zip(unique, counts)))
