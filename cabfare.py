3# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 16:10:29 2019

@author: pc
"""

#importing Libraries
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor


#from fancyimpute import KNN

# Setting up new woorking directory
os.chdir("C:\\Users\\pc\\Desktop\\R\\projects\\Cab care project")
#Checking current directory
os.getcwd()

#importing data
train = pd.read_csv("train_cab.csv")
test = pd.read_csv("test.csv")
# Checking data dimenssions
train.shape
test.shape
# No. of rows in data
train.shape[0]
test.shape[0]
# No. of columns
train.shape[1]
test.shape[1]
# name of columns
list(train)
list(test)
# data detail
train.info()
test.info()

# Converting data into required format
data = [train, test]
for i in data:
    i['pickup_datetime'] = pd.to_datetime(i['pickup_datetime'], errors='coerce')

#train['fare_amount'].astype(float)
train['fare_amount'] = np.where(train['fare_amount']== "430-", 430, train['fare_amount'] )
train['fare_amount'] = train['fare_amount'].astype(float)

############################################## Missing Value Analysis
train.isnull().sum()
test.isnull().sum()     # 0 missing value

# we are getting 1 missing value in pickup_datetime, we will drop that observation.
np.where(train['pickup_datetime'].isnull())
train = train.dropna(subset = ['pickup_datetime'], how = 'all') 

# filling its values by mode, Becouse passenger_count is categorical variable
train['passenger_count'] = train['passenger_count'].fillna(train['passenger_count'].mode()[0])

# We will convert Passenger_count variable into object type, becouse it is categorical variable
train['passenger_count']=train['passenger_count'].round().astype('object').astype('category')
test['passenger_count']=test['passenger_count'].round().astype('object').astype('category')

missing_val = pd.DataFrame(train.isnull().sum())
#Reset index
missing_val = missing_val.reset_index()
# Reanaming variables
missing_val = missing_val = missing_val.rename(columns = {'index': 'Variables', 0: 'Missing_percentage'})
#Calculate percentage
missing_val['Missing_percentage'] = (missing_val['Missing_percentage']/len(train))*100


# Best method
#actual value= 10
#mean=  15.041185637700874
#median= 8.5

# we will a value to replace na in fare_amount variable
train['fare_amount'].loc[100] = np.nan
#Mean
train['fare_amount'] = train['fare_amount'].fillna(train['fare_amount'].mean())
#Median
train['fare_amount'] = train['fare_amount'].fillna(train['fare_amount'].median())
# we have find median as best method to fill null values
train.fillna(value = train.median(), inplace= True)

############################# Outliers analysis
# we will use cap filling to replace outliers

plt.boxplot(train["fare_amount"])
plt.boxplot(train["pickup_latitude"])
plt.boxplot(train['pickup_longitude'])
plt.boxplot(train["dropoff_latitude"])
plt.boxplot(train['dropoof_longitude'])

def outlier_detect(df):
    for i in df.describe().columns:
        q1=df.describe().at["25%",i]
        q3=df.describe().at["75%",i]
        IQR=(q3-q1)
        lb=(q1-1.5*IQR)
        ub=(q3+1.5*IQR)
        x=np.array(df[i])
        p=[]
        for j in x:
             if j<lb:
                p.append(lb)
             elif j>ub:
                p.append(ub)
             else:
                p.append(j)
        df[i]=p
    return(df)

outlier_detect(train)
outlier_detect(test)

################################# Feature Engineering
#1. Feautre Engineering for fare_amount variable

#Removing values which are not within desired range(outlier) depending upon basic understanding of dataset.
# In fare_amount values which are lesser then 1 dont have any significance in data
Counter(train['fare_amount']<1)
train = train.drop(train[train['fare_amount']<1].index, axis=0)

#2. Feature engineering for pickup_datetime variable

#lets create a function to get important features from pickup_datetime variable in train and test datasets
#train = pd.concat((pickup_datetime, train), axis= 1)

data = [train,test]
for i in data:
    i["year"] = i["pickup_datetime"].apply(lambda row: row.year)
    i["month"] = i["pickup_datetime"].apply(lambda row: row.month)
    i["day_of_month"] = i["pickup_datetime"].apply(lambda row: row.day)
    i["day_of_week"] = i["pickup_datetime"].apply(lambda row: row.dayofweek)
    i["hour"] = i["pickup_datetime"].apply(lambda row: row.hour)


#3. Feature engineering for passenger_count variable
train['passenger_count']=train['passenger_count'].astype('int')

test['passenger_count'].unique()
train['passenger_count'].unique()

train['passenger_count']=train['passenger_count'].round().astype('object')
train.std()

for i in range(4,11):
    print('passenger_count above' +str(i)+'={}'.format(sum(train['passenger_count']>i)))

#so 20 observations of passenger_count is consistenly above from 6,7,8,9,10 passenger_counts, let's check them.
Counter(train['passenger_count']>6)
#Also we need to see if there are any passenger_count<1
Counter(train['passenger_count']<1)

#passenger_count variable conatins values which are equal to 0.
#we will remove those 0 values.
#Also, We will remove 20 observation which are above 6 value because a cab cannot hold these number of passengers.

train = train.drop(train[train['passenger_count']>6].index, axis=0)
train = train.drop(train[train['passenger_count']<1].index, axis=0)
train.std()
train['passenger_count']=train['passenger_count'].astype('int')
train['passenger_count'].unique()

#4. Feature engineering for latitude and logitude

#Latitudes range from -90 to 90.Longitudes range from -180 to 180. 
#Removing which does not satisfy these ranges
print('pickup longitude above 180={}'.format(sum(train['pickup_longitude']>180)))
print('pickup_longitude below -180={}'.format(sum(train['pickup_longitude']<-180)))
print('pickup_latitude above 90={}'.format(sum(train['pickup_latitude']>90)))
print('pickup_latitude below -90={}'.format(sum(train['pickup_latitude']<-90)))
print('dropoff_longitude above 180={}'.format(sum(train['dropoff_longitude']>180)))
print('dropoff_longitude below -180={}'.format(sum(train['dropoff_longitude']<-180)))
print('dropoff_latitude below -90={}'.format(sum(train['dropoff_latitude']<-90)))
print('dropoff_latitude above 90={}'.format(sum(train['dropoff_latitude']>90)))


#There's only one outlier which is in variable pickup_latitude.So we will remove it.
#train = train.drop(train[train['pickup_latitude']>90].index, axis=0)

#Also we will see if there are any values equal to 0.
for i in ['pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude']:
    print(i,'equal to 0={}'.format(sum(train[i]==0)))

#there are values which are equal to 0. we will remove them.
for i in ['pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude']:
    train = train.drop(train[train[i]==0].index, axis=0)

train.shape
train.info()


### Now let's calculate trip distance from picup and dropoff latitude and longitude
def trip_distance(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    km = 6371 * c  
    return km

train['trip_distance']=trip_distance(train['pickup_longitude'],train['pickup_latitude'],
                                     train['dropoff_longitude'],train['dropoff_latitude'])

test['trip_distance']=trip_distance(test['pickup_longitude'],test['pickup_latitude'],
                                     test['dropoff_longitude'],test['dropoff_latitude'])

###we will remove the rows whose distance value is zero
Counter(train['trip_distance']==0)
train = train.drop(train[train['trip_distance']== 0].index, axis=0)
train.shape

Counter(test['trip_distance']==0)
test = test.drop(test[test['trip_distance']== 0].index, axis=0)
train.shape

#Now we will plot a scatter plot
plt.xlabel("trip Distance")
plt.ylabel("Fare Amount")
plt.scatter(x=train['trip_distance'],y=train['fare_amount'])
plt.title("Trip Distance vs Fare Amount")

train.describe()

df=train.copy()
# train=df.copy()

# # Data Visualization :

# Visualization of following:
# 
# 1. Number of Passengers effects the the fare
# 2. Pickup date and time effects the fare
# 3. Day of the week does effects the fare
# 4. Distance effects the fare

plt.figure(figsize=(20,10))
sns.countplot(train['year'])
# plt.savefig('year.png')
plt.figure(figsize=(20,10))
sns.countplot(train['month'])


# Count plot on passenger count
plt.figure(figsize=(15,7))
sns.countplot(x="passenger_count", data=train)

#Relationship beetween number of passengers and Fare

plt.figure(figsize=(15,7))
plt.scatter(x=train['passenger_count'], y=train['fare_amount'], s=10)
plt.xlabel('No. of Passengers')
plt.ylabel('Fare')
plt.show()


#Relationship between date and Fare
plt.figure(figsize=(15,7))
plt.scatter(x=train['day_of_month'], y=train['fare_amount'], s=10)
plt.xlabel('Date')
plt.ylabel('Fare')
plt.show()
# day_of_month is not much significance in dataset.
# 
plt.figure(figsize=(20,10))
sns.countplot(train['hour'])
plt.show()


# Lowest cabs at 5 AM and highest at and around 7 PM i.e the office rush hours

#Relationship between Time and Fare
plt.figure(figsize=(15,7))
plt.scatter(x=train['hour'], y=train['fare_amount'], s=10)
plt.xlabel('Hour')
plt.ylabel('Fare')
plt.show()

# From the above plot We can observe that the cabs taken at 7 am and 23 Pm are the costliest. 
# Hence we can assume that cabs taken early in morning and late at night are costliest

#impact of Day of week on the number of cab rides
plt.figure(figsize=(15,7))
sns.countplot(x="day_of_week", data=train)


# Observation :
# The day of the week does not seem to have much influence on the number of cabs ride


#We will remove the variables which were used to feature engineer new variables
train=train.drop(['pickup_datetime','pickup_longitude', 'pickup_latitude',
'dropoff_longitude', 'dropoff_latitude'],axis=1)

test=test.drop(['pickup_datetime','pickup_longitude', 'pickup_latitude',
'dropoff_longitude', 'dropoff_latitude'],axis=1)
  

############################## Feature Selection

# Calculation of correlation between numerical variables
num_var=['fare_amount','trip_distance']
df_num = train.loc[:,num_var]

corr = df_num.corr()
print(corr)

# plotiing the heatmap
f, ax = plt.subplots(figsize=(7, 5))
sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),square=True, ax=ax)
plt.show()
## we can say from the plot that both variable are correlated to each other

################################################## Anova test
from statsmodels.formula.api import ols
import statsmodels.api as sm
model = ols('fare_amount ~ C(day_of_week)+C(passenger_count)+C(day_of_month)+C(year)+C(hour)',data=train).fit()
aov_table = sm.stats.anova_lm(model)
aov_table
# we are getting two values day_of_week and day_of_month values higher then 0.05, so will drop them

########################################### Multicollinearity Test
#VIF is always greater or equal to 1.
#if VIF is 1 --- Not correlated to any of the variables.
#if VIF is between 1-5 --- Moderately correlated.
#if VIF is above 5 --- Highly correlated.
#If there are multiple variables with VIF greater than 5, only remove the variable with the highest VIF.

# Detecting and Removing Multicollinearity 
# use statsmodels library to calculate VIF
# Import VIF function from statmodels Library
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Get variables for which to compute VIF and add intercept term:

X = train[['passenger_count', 'year',
               'month', 'day_of_month', 'day_of_week', 'hour','trip_distance']].dropna() #subset the dataframe
X ['Intercept'] = 1

# Compute and view VIF:

vif = pd.DataFrame()           # Create an empty dataframe
vif["Variables"] = X.columns   # Add "Variables" column to empty dataframe
vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

# View results using print
print(vif)

# from the results,we will just drop 2 variables=

train=train.drop(['day_of_week','day_of_month'], axis=1)
test=test.drop(['day_of_week','day_of_month'], axis=1)  


#Normality check of training data is uniformly distributed or not-

for i in ['fare_amount', 'trip_distance']:
    print(i)
    sns.distplot(train[i],bins='auto',color='green')
    plt.title("Distribution for Variable "+i)
    plt.ylabel("Density")
    plt.show()

# data is normally distributed

######################################## Model development

# divided into independent (x) and dependent variables (y)
x= train.iloc[:,1:6]
x.shape
y =train.iloc[:,0]
y

 # Splitting the data into training and test sets
x_train, x_test,y_train, y_test =train_test_split(x,y,test_size=.2, random_state =100)
print(train.shape, x_train.shape, x_test.shape,y_train.shape,y_test.shape)

### Linear Regression

# linear regression using sklearn

lm =LinearRegression()
lm= lm.fit(x_train,y_train)
# coefficients
lm.coef_
# To store coefficients in a data frame along with their respective independent variables
coefficients=pd.concat([pd.DataFrame(x_train.columns),pd.DataFrame(np.transpose(lm.coef_))], axis = 1)
print(coefficients)

# intercept
lm.intercept_

#prediction on train data
pred_train = lm.predict(x_train)

#prediction on test data
pred_test = lm.predict(x_test)

##calculating RMSE for test data
RMSE_test = np.sqrt(mean_squared_error(y_test, pred_test))

##calculating RMSE for train data
RMSE_train= np.sqrt(mean_squared_error(y_train, pred_train))

print("Root Mean Squared Error For Training data = "+str(RMSE_train))  #3.2878183375029786
print("Root Mean Squared Error For Test data = "+str(RMSE_test))  #3.078645997656653

#calculate R^2 for train data
r2_score(y_train, pred_train) #0.6393978978272477
# r2 foe test data
r2_score(y_test, pred_test) #0.675381984761757

###### Decision tree

#Decision tree for regression
fit_DT = DecisionTreeRegressor(max_depth=6).fit(x_train, y_train)

#prediction on train data
pred_DT = fit_DT.predict(x_train)
#Apply model on test data
predictions_DT = fit_DT.predict(x_test)

#calculate R^2 for train data
r2_score(y_train, pred_DT) #0.7030262920691919
# r2 for test data
r2_score(y_test, predictions_DT) #0.7155614076403296

# RMSE for both data
RMSE_train=np.sqrt(mean_squared_error(pred_DT,y_train))
RMSE_test=np.sqrt(mean_squared_error(predictions_DT,y_test))

print("RMSE of train data = ", RMSE_train)  #2.983683049406448
print("RMSE of test data = ",RMSE_test)   # 2.881825667387774


#### Random Forest

fit_RF = RandomForestRegressor(n_estimators = 500).fit(x_train,y_train)

#prediction on train data
pred_RF = fit_RF.predict(x_train)
#Apply model on test data
predictions_RF = fit_RF.predict(x_test)

#calculate R^2 for train data
r2_score(y_train, pred_RF) # 0.9529716271574351
# r2 foe test data
r2_score(y_test, predictions_RF)  # 0.6948525495517299

# RMSE for both data
RMSE_train=np.sqrt(mean_squared_error(pred_RF,y_train))
RMSE_test=np.sqrt(mean_squared_error(predictions_RF,y_test))

print("RMSE of train data = ", RMSE_train)  #1.187336073286684
print("RMSE of test data = ",RMSE_test)      #2.9848899080585043


# ### As we got best Accuracy with Decision Tree Model we will use this Model to predict Fare

# test data
test.describe()
test.shape
# prediction on test data using decision tree mode;
predicted_fare=fit_DT.predict(test)
# Saving predicted fare in test data
test['predicted_fare']=predicted_fare

test.head(10)
# saving test data in our memory
test.to_csv("test_predicted.csv",index=False)
