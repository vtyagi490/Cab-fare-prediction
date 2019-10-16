# Cab Fare prediction

rm(list=ls())
setwd("C:\\Users\\pc\\Desktop\\R\\projects\\Cab care project")

#Load Libraries
x = c("ggplot2", "corrgram", "DMwR", "caret", "randomForest", "unbalanced", "dummies", "e1071", "Information",
      "MASS", "rpart", "gbm", "ROSE", 'sampling', 'DataCombine', 'inTrees', 'dplyr')

#install.packages(x)
lapply(x, require, character.only = TRUE)
rm(x)


# The details of data attributes in the dataset are as follows:
# pickup_datetime - timestamp value indicating when the cab ride started.
# pickup_longitude - float for longitude coordinate of where the cab ride started.
# pickup_latitude - float for latitude coordinate of where the cab ride started.
# dropoff_longitude - float for longitude coordinate of where the cab ride ended.
# dropoff_latitude - float for latitude coordinate of where the cab ride ended.
# passenger_count - an integer indicating the number of passengers in the cab ride.

# loading datasets
train = read.csv("train_cab.csv", header = T)
test = read.csv("test.csv")

# Structure of data
str(train)
str(test)
summary(train)
summary(test)
head(train,5)
head(test,5)
# Check class of the data
class(train)

#Check the dimensions(no of rows and no of columns)
dim(train)

#Check names of dataset(no need of renaming variables)
names(train)


# Let's Check for data types of train data:
sapply(train, class)
str(train)

# Let's Check for data types of train data:
sapply(train, class)
str(train)

#############                              Exploratory Data Analysis                    #######################

# In train data observed that fare_amount and pickup_datetime variables are of Factor type. 
# and passenger_count variable of numeric type
# So, Need to convert fare_amount datatype to 'numeric' & pickup_datetime data type to 'datatime' format.
# Passenger_count datatype to integer datatype

train$fare_amount = as.numeric(as.character(train$fare_amount))
class(train$fare_amount)# Data type After conversion


# Convert Passeneger_count data type from numeric to integer type:
class(train$passenger_count)
train$passenger_count = as.integer(train$passenger_count)
class(train$passenger_count)

# Convert pickup_datetime data type from factor to datetime

train$pickup_datetime <- as.POSIXct(strptime(train$pickup_datetime, "%Y-%m-%d %H:%M:%S"))
test$pickup_datetime <- as.POSIXct(strptime(test$pickup_datetime, "%Y-%m-%d %H:%M:%S"))

str(train$pickup_datetime)

head(train)

summary(train) # There is one observation in pickup_datetime which is not in correct format need to delete it.

# Check observations which are not formatted  correctly in pickup_datetime variable 

train[is.na(strptime(train$pickup_datetime,format="%Y-%m-%d %H:%M:%S")),]

# Remove observation which are not in correct datetime format :In our case 1 observation found. 

sum(is.na(train$pickup_datetime))
train <- train[-c(1328),]
dim(train)



#############                        Missing Value Analysis                  #############
missing_val = data.frame(apply(train,2,function(x){sum(is.na(x))}))
missing_val$Columns = row.names(missing_val)
names(missing_val)[1] =  "Missing_percentage"
missing_val$Missing_percentage = (missing_val$Missing_percentage/nrow(train)) * 100
missing_val = missing_val[order(-missing_val$Missing_percentage),]
row.names(missing_val) = NULL
missing_val = missing_val[,c(2,1)]
# percentage of missing values is lower then 30 %.

##As passenger_count is categorical variable we will impute it using mode
###Mode Method
train$passenger_count[is.na(train$passenger_count)] =as.data.frame(mode(train$passenger_count))

df=train
#train=df

#actual value=10.9
#mean=15.0526
#median=8.5

train[1000,1] = NA

## we will impute missing value of fare_amount by using mean or median method
####Mean Method
train$fare_amount[is.na(train$fare_amount)] = mean(train$fare_amount, na.rm = T)

####Median Method
train$fare_amount[is.na(train$fare_amount)] = median(train$fare_amount, na.rm = T)


sum(is.na(train))
str(train)
summary(train)

#####################                        Outlier Analysis                 ##################

# #Plot boxplot to visualize Outliers

#lets check the NA's  in train data
sum(is.na(train$fare_amount))

cnames = colnames(train[,c("fare_amount","pickup_longitude","pickup_latitude","dropoff_longitude","dropoff_latitude")])

for (i in 1:length(cnames))
{
  assign(paste0("gn",i), ggplot(aes_string(y = cnames[i]), data = train)+
           stat_boxplot(geom = "errorbar", width = 0.5) +
           geom_boxplot(outlier.colour="red", fill = "grey" ,outlier.shape=18,
                        outlier.size=1, notch=FALSE) +
           theme(legend.position="bottom")+
           labs(y=cnames[i])+
           ggtitle(paste("Box plot for",cnames[i])))
}
gridExtra::grid.arrange(gn1,gn3,gn2,gn4,ncol=2)

# dropping the outliers

for (i in cnames)
{
  val = train[,i][train[,i] %in% boxplot.stats(train[,i])$out]
  train = train[which(!train[,i] %in% val),]
}

#lets check the NA's  in test data
sum(is.na(test))

cname = colnames(test[,c("pickup_longitude","pickup_latitude","dropoff_longitude","dropoff_latitude")])

for (i in 1:length(cname))
{
  assign(paste0("gn",i), ggplot(aes_string(y = cname[i]), data = test)+
           stat_boxplot(geom = "errorbar", width = 0.5) +
           geom_boxplot(outlier.colour="red", fill = "grey" ,outlier.shape=18,
                        outlier.size=1, notch=FALSE) +
           theme(legend.position="bottom")+
           labs(y=cname[i])+
           ggtitle(paste("Box plot for",cname[i])))
}
gridExtra::grid.arrange(gn1,gn3,gn2,gn4,ncol=2)

# dropping the outliers

for (i in cname)
{
  val = test[,i][test[,i] %in% boxplot.stats(test[,i])$out]
  test = test[which(!test[,i] %in% val),]
}


#############                        Feature Engineering                  #############

### Removing values which are not within desired range(outlier) depending upon basic understanding of dataset.

# 1.Fare amount has a negative value, which doesn't make sense. A price amount cannot be -ve and also cannot be 0. So we will remove these fields.
train[which(train$fare_amount < 1 ),]
nrow(train[which(train$fare_amount < 1 ),])
train = train[-which(train$fare_amount < 1 ),]

#2.Passenger_count variable
for (i in seq(4,11,by=1)){
  print(paste('passenger_count above ' ,i,nrow(train[which(train$passenger_count > i ),])))
}
# so some observations of passenger_count is consistenly above from 6,7,8,9,10 passenger_counts, let's check them.
train[which(train$passenger_count > 6 ),]
# Also we need to see if there are any passenger_count==0
train[which(train$passenger_count <1 ),]
nrow(train[which(train$passenger_count <1 ),])

# We will remove these observation which are above 6 value because a cab cannot hold these number of passengers.
train = train[-which(train$passenger_count < 1 ),]
train = train[-which(train$passenger_count > 6),]

# converting passenger_count as categorical variable
train$passenger_count = as.integer(train$passenger_count)
train$passenger_count = as.factor(train$passenger_count)
test$passenger_count = as.factor(test$passenger_count)

# 3.Feature Engineering for timestamp variable
# we will derive new features from pickup_datetime variable
# new features will be year,month,day_of_week,hour
#Convert pickup_datetime from factor to date time
train$day = as.factor(format(train$pickup_datetime,"%d"))
train$weekday = as.factor(format(train$pickup_date,"%u"))# Monday = 1
train$month = as.factor(format(train$pickup_date,"%m"))
train$year = as.factor(format(train$pickup_date,"%Y"))
pickup_time = strptime(train$pickup_datetime,"%Y-%m-%d %H:%M:%S")

train$hour = as.factor(format(pickup_time,"%H"))

#Add same features to test set
test$day = as.factor(format(test$pickup_datetime,"%d"))
test$weekday = as.factor(format(test$pickup_date,"%u"))# Monday = 1
test$month = as.factor(format(test$pickup_date,"%m"))
test$year = as.factor(format(test$pickup_date,"%Y"))
pickup_time = strptime(test$pickup_datetime,"%Y-%m-%d %H:%M:%S")
test$hour = as.factor(format(pickup_time,"%H"))


train = subset(train,select = -c(pickup_datetime,pickup_datetime))
test = subset(test,select = -c(pickup_datetime,pickup_datetime))

#Longitude range----(-180 to 180)
#Latitude range----(-90 to 90)
# Check observations having pickup longitute and pickup latitute out the range in train dataset.

train[train$pickup_longitude <  -180,]
train[train$pickup_longitude >  180,]
train[train$pickup_latitude <  -90,]
train[train$pickup_latitude >  90,]


# Check observations having dropoff longitute and dropoff latitute out the range in train dataset.

train[train$dropoff_longitude <  -180,]
train[train$dropoff_longitude >  180,]
train[train$dropoff_latitude <  -90,]
train[train$dropoff_latitude >  90,]


# Dropping the observations which are outof range in train dataset:

train<-  filter (train,pickup_longitude >  -180)
train<-  filter (train,pickup_longitude <  180)
train<-  filter (train,pickup_latitude >  -90)
train<-  filter (train,pickup_latitude <  90)
dim(train)

train<-  filter (train,dropoff_longitude >  -180)
train<-  filter (train,dropoff_longitude <  180)
train<-  filter (train,dropoff_latitude >  -90)
train<-  filter (train,dropoff_latitude <  90)
dim(train)

#Longitude range----(-180 to 180)
#Latitude range----(-90 to 90)
# Check observations having pickup longitute and pickup latitute out the range in train dataset.

test[test$pickup_longitude <  -180,]
test[test$pickup_longitude >  180,]
test[test$pickup_latitude <  -90,]
test[test$pickup_latitude >  90,]


# Check observations having dropoff longitute and dropoff latitute out the range in test dataset.

test[test$dropoff_longitude <  -180,]
test[test$dropoff_longitude >  180,]
test[test$dropoff_latitude <  -90,]
test[test$dropoff_latitude >  90,]


# Dropping the observations which are outof range in test dataset:

test<-  filter (test,pickup_longitude >  -180)
test<-  filter (test,pickup_longitude <  180)
test<-  filter (test,pickup_latitude >  -90)
test<-  filter (test,pickup_latitude <  90)
dim(test)

test<-  filter (test,dropoff_longitude >  -180)
test<-  filter (test,dropoff_longitude <  180)
test<-  filter (test,dropoff_latitude >  -90)
test<-  filter (test,dropoff_latitude <  90)


# Also we will see if there are any values equal to 0.
nrow(train[which(train$pickup_longitude == 0 ),])
nrow(train[which(train$pickup_latitude == 0 ),])
nrow(train[which(train$dropoff_longitude == 0 ),])
nrow(train[which(train$pickup_latitude == 0 ),])


### Now let's calculate trip distance from picup and dropoff latitude and longitude
## Haversine
trip_distance = function(lon1, lat1, lon2, lat2){
  # convert decimal degrees to radians
  lon1 = lon1 * pi / 180
  lon2 = lon2 * pi / 180
  lat1 = lat1 * pi / 180
  lat2 = lat2 * pi / 180
  # haversine formula
  dlon = lon2 - lon1
  dlat = lat2 - lat1
  a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
  c = 2 * atan2(sqrt(a), sqrt(1-a))
  km = 6367 * c
  return(km)
}

## Calculating trip_distance for train data
train$trip_distance=trip_distance(train$pickup_longitude,train$pickup_latitude,
                                  train$dropoff_longitude,train$dropoff_latitude)


## Calculating trip_distance for test data
test$trip_distance=trip_distance(test$pickup_longitude,test$pickup_latitude,
                                 test$dropoff_longitude,test$dropoff_latitude)

# We will remove the variables which were used to feature engineer new variables
train = subset(train,select = -c(pickup_longitude,pickup_latitude,dropoff_longitude,dropoff_latitude))
test = subset(test,select = -c(pickup_longitude,pickup_latitude,dropoff_longitude,dropoff_latitude))

## now lets look at the summary of data 
summary(train)

##Now remove the trip_distance having value less than 0, becouse these values are not useful
train=subset(train, trip_distance>0)

df1=train
# train=df1

#########################  Visualization   ######################################
# Visualization between fare_amount and Hours.
ggplot(data = train, aes(x = hour, y = fare_amount, fill = hour))+
  geom_bar(stat = "identity")+
  labs(title = "Fare Amount Vs.Hour", x = "Hours", y = "Fare Amount",subtitle = "Bi - Variate Analysis", 
       caption = "(Observation : Rides taken during 6 pm to 11 pm gives highest fare_amount.)")+
  theme(plot.title = element_text(hjust = 0.5, face = "bold"))+
  theme(axis.text.x = element_text( color="Red", size=10, angle=0))+
  theme(axis.text.y = element_text( color="blue", size=10, angle=0))

# Visualization between fare_amount and day.
ggplot(data = train, aes(x = day, y = fare_amount, fill = day))+
  geom_bar(stat = "identity")+
  labs(title = "Fare Amount Vs. Day", x = "Day", y = "Fare Amount",subtitle = "Bi - Variate Analysis", 
       caption = "(Observation : Rides taken during midweeks gives highest fare_amount.)")+
  theme(plot.title = element_text(hjust = 0.5, face = "bold"))+
  theme(axis.text.x = element_text( color="Red", size=10, angle=0))+
  theme(axis.text.y = element_text( color="blue", size=10, angle=0))

# Visualization between fare_amount and weekday.
ggplot(data = train, aes(x = weekday,y = fare_amount, fill = weekday))+
  geom_bar(stat = "identity")+
  labs(title = "Fare Amount Vs. weekday", x = "weekday", y = "Fare Amount",subtitle = "Bi - Variate Analysis",
       caption = "(Observation : Thursday to Saturday rides has the highest fare_amount.)")+
  theme(plot.title = element_text(hjust = 0.5, face = "bold"))+
  theme(axis.text.x = element_text( color="Red", size=10, angle=0)) +
  theme(axis.text.y = element_text( color="Brown", size=10, angle=0))

# Visualization between fare_amount and years.
ggplot(data = train, aes(x = year, y = fare_amount, fill= year))+
  geom_bar(stat = "identity")+ 
  labs(title = "Fare Amount V/s years",  x = "Years", y = "Fare Amount", subtitle = "Bi - Variate Analysis", 
       caption = "(Observation : In year 2013 there were rides which got high fare_amount)") +
  theme(plot.title = element_text(hjust = 0.5, face = "bold"))+
  theme(axis.text.x = element_text( color="Red", size=10, angle=0))+
  theme(axis.text.y = element_text( color="Brown", size=10, angle=0))

# Visualization between fare_amount and months.
#col <- rainbow(ncol(train))
ggplot(train, aes(x = month, y = fare_amount, fill= month))+ 
  geom_bar(stat = "identity")+
  labs(title = " Fare Amount V/s. Month", x = "Months", y = "Fare Amount", subtitle = "Bi - Variate Analysis",
       caption = "(Observation: Month May collects the highest fare_amount)")+
  theme(plot.title = element_text(hjust = 0.5, face = "bold"))+
  theme(axis.text.x = element_text( color="Red", size=10, angle = 0))+
  theme(axis.text.y = element_text( color="Brown", size=10, angle = 0))


################                             Feature selection                 ###################
numeric_index = sapply(train,is.numeric) #selecting only numeric

numeric_data = train[,numeric_index]

cnames = colnames(numeric_data)
#Correlation analysis for numeric variables
corrgram(train[,numeric_index],upper.panel=panel.pie, main = "Correlation Plot")

#ANOVA for categorical variables with target numeric variable
aov_results = aov(fare_amount ~ passenger_count + hour + weekday + month + year+ day,data = train)
summary(aov_results)

# pickup_weekday has p value greater than 0.05 
train = subset(train,select = -c(weekday, day))
test = subset(test,select = -c(weekday, day))

#########################      Feature Scaling                  #######################
qqnorm(train$fare_amount)
hist(train$fare_amount)

hist(train$trip_distance)
qqnorm(train$trip_distance)

#data is normally distributed, no need of normalization


#################### Splitting train into train and validation subsets ###################
set.seed(1000)
tr.idx = createDataPartition(train$fare_amount,p=0.80,list = FALSE) # 80% in trainin and 20% in Validation Datasets
train_data = train[tr.idx,]
test_data = train[-tr.idx,]

rmExcept(c("test","train","df",'df1','df2','df3','test_data','train_data','test_pickup_datetime'))
###################Model Selection################
#Error metric used to select model is RMSE


#############            Linear regression               #################
lm_model = lm(fare_amount ~.,data=train_data)

summary(lm_model)  #R-squared:  0.7129
str(train_data)
plot(lm_model$fitted.values,rstandard(lm_model),main = "Residual plot",
     xlab = "Predicted values of fare_amount",
     ylab = "standardized residuals")


lm_predictions = predict(lm_model,test_data[,2:6])

qplot(x = test_data[,1], y = lm_predictions, data = test_data, color = I("blue"), geom = "point")

regr.eval(test_data[,1],lm_predictions)
#mae       mse      rmse      mape 
#1.4530745 4.2994675 2.0735157 0.1791255  

#############                             Decision Tree            #####################

Dt_model = rpart(fare_amount ~ ., data = train_data, method = "anova")

summary(Dt_model)
#Predict for new test cases
predictions_DT = predict(Dt_model, test_data[,2:6])

qplot(x = test_data[,1], y = predictions_DT, data = test_data, color = I("blue"), geom = "point")

regr.eval(test_data[,1],predictions_DT)
#mae       mse      rmse      mape 
#1.6439027 5.0444347 2.2459819 0.2129453  


#############                             Random forest            #####################
rf_model = randomForest(fare_amount ~.,data=train_data)

summary(rf_model)

rf_predictions = predict(rf_model,test_data[,2:6])

qplot(x = test_data[,1], y = rf_predictions, data = test_data, color = I("blue"), geom = "point")

regr.eval(test_data[,1],rf_predictions)
#mae      mse     rmse     mape 
#1.668754 5.162626 2.272141 0.217559 

### As we got best Accuracy with Linear Regression Model we will use this Model to predict Fare
summary(test)

predicted_Fare=predict(lm_model,test)

test$Predicted_fare=predicted_Fare

write.csv(test, "predicted_test_R.csv", row.names = F)
