#Data preprocessing
#importing libraries
import numpy as np#contains mathematical tools
import matplotlib.pyplot as plt
import pandas as pd
#importing datasets

dataset = pd.read_csv('Data.csv')

#creating independant and dependant vectors
#iloc will contain all rows and columsn except last column
X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,3].values

from sklearn.preprocessing import Imputer
imputer  = Imputer(missing_values  = 'NaN', strategy  = 'mean', axis = 0)
imputer = imputer.fit(X[:,1:3])#Ti is fitting all rows and 1st and 2nd column but 3 is confusing stil it is 2nd column
X[:,1:3] = imputer.transform(X[:,1:3])

#encoding the categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

labelEncoder_X  = LabelEncoder()
X[:,0] = labelEncoder_X.fit_transform(X[:,0]) 

#creating dummies
onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()

#Since the purchase data is dependent it will know and we will not have to use onehotencoder
labelEncoder_Y = LabelEncoder()
Y  = labelEncoder_Y.fit_transform(Y)

#spliting data into train and test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size = 0.2,random_state = 0)

#feature scaling 
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()

X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)



