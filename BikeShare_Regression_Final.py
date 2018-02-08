#https://elitedatascience.com/python-machine-learning-tutorial-scikit-learn
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn import neighbors
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.feature_extraction import DictVectorizer
from sklearn.externals import joblib

def wrangle(data):
    # engineer date and time features - this is specific to the "BikeShare" contest
    temp = pd.DatetimeIndex(data['datetime'])
    data['date'] = temp.date
    data['time'] = temp.time

    # create a feature called hour
    data['hour'] = pd.to_datetime(data.time, format="%H:%M:%S")
    data['hour'] = pd.Index(data['hour']).hour
    
    # there appears to be a general increase in rentals over time, so days from start should be captured
    data['dateDays'] = (data.date - data.date[0]).astype('timedelta64[D]')
    
    # create a categorical feature for day of the week (0=Monday to 6=Sunday)
    data['dayofweek'] = pd.DatetimeIndex(data.date).dayofweek
    
    # create binary features which show if day is Saturday/Sunday
    data['Saturday']=0
    data.Saturday[data.dayofweek==5]=1
    
    data['Sunday']=0
    data.Sunday[data.dayofweek==6]=1

    dataRel = data.drop(['datetime','date','time','dayofweek'], axis=1)
    
    return(dataRel)

def OneHot(X):
    
    X_CAT = X.drop(CATEGORICAL,axis=1)
    X_CONT = X.drop(CONTINUOUS ,axis=1)

    enc = preprocessing.OneHotEncoder()
    enc.fit(X_CAT)
    onehotlabels = enc.transform(X_CAT).toarray()

    X_TOTAL = np.concatenate([onehotlabels, X_CONT], axis=1)
    
    return(X_TOTAL)
    
    
if __name__ == "__main__":
    ##import data to Pandas Frame:
    dataset_url = 'bike_rental_all.csv'
    data = pd.read_csv(dataset_url)

    data = wrangle(data)

    #data looks like this:
    #datetime,month,day,hour,dayofweek,season,holiday,workingday,weather,temp,humidity,windspeed,casual,registered,counted
    #2011-09-03 00:00:00,9,3,0,5,3,0,0,2,26.24,73,7.0015,22,65,87
    #2012-08-13 14:00:00,8,13,14,0,3,0,1,1,32.8,33,7.0015,85,163,248
    
    ##seperate training and target data (keywords taken from data file above):
    y = data.counted
    #X = data.drop(['counted','datetime','casual','registered'], axis=1)
    X = data.drop(['counted', 'casual', 'registered'], axis=1)
    CATEGORICAL = ['temp','humidity','windspeed','dateDays','month','day','hour','season','workingday']
    CONTINUOUS = ['Sunday','Saturday','weather','holiday']
    #label encode categorical values
    X = OneHot(X)
    
    #create list for models - 
    models = []
    models.append(('RFR', RandomForestRegressor()))
    #models.append(('GBR', GradientBoostingRegressor()))
    #models.append(('ABR', AdaBoostRegressor()))
    #models.append(('DTR', DecisionTreeRegressor()))
    #models.append(('LNR', LinearRegression()))
    #models.append(('LGR', LogisticRegression()))
    #models.append(('KNN', neighbors.KNeighborsRegressor()))
    
    
    #create training and test data:
    for name, model in models:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=123)
        #create pipeline - normalize data and select method
        
        pipeline = make_pipeline(preprocessing.StandardScaler(), model)
        #define hyperparamaters to tune
        #hyperparameters = { 'randomforestregressor__max_features' : ['auto', 'sqrt', 'log2'],'randomforestregressor__max_depth': [None, 5, 3, 1]}
        hyperparameters = {}
        #carry out cross validation pipeline (tests training data against all hyperparameter permutations)
        clf = GridSearchCV(pipeline, hyperparameters, cv=10)
     
        # Fit and tune model
        clf.fit(X_train, y_train)
        #predict target against test data
        y_pred = clf.predict(X_test)
        #test prediction against actual test data
        print (name)
        print("R squared value: ")
        #print (r2_score(y_test, y_pred))
        #print("RMS value: ")
        #print (mean_squared_error(y_test, y_pred))
        #print("  ")