import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

dataset = pd.read_csv("C:\\Users\\Gustavo\\Documents\\Big data e Cloud\\Machine_Learning_AZ_Template_Folder\\Machine Learning A-Z Template Folder\\Part 1 - Data Preprocessing\\Data_Preprocessing\\Data.csv")

# Independent values matrix creation
X = dataset.iloc[:, :-1].values

# Dependent variables vector creation
y = dataset.iloc[:, 3].values

# Creation of the Imputer object, strategy is how the values will be replaced can be mean, median or most frequent
# axis is the direction (0 to columns and 1 to rows)
imp = Imputer(missing_values="NaN", strategy="mean", axis=0)

# Fitting the imputer on de X matrix
imp = imp.fit(X[:, 1:3])
X[:, 1:3] = imp.transform(X[:, 1:3])

# Transforming categorical values(X)
labelEncoderX = LabelEncoder()
X[:, 0] = labelEncoderX.fit_transform(X[:, 0])

# OneHotEncoder is used to prevent ordering between in Country column
hotEncoder = OneHotEncoder(categorical_features=[0])
X = hotEncoder.fit_transform(X).toarray()

# Transforming categorical values (y)
labelEncoderY = LabelEncoder()
y = labelEncoderY.fit_transform(y)

# Split data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75, test_size=0.25)

# Feature Scaling of X
scale_X = StandardScaler()
X_train = scale_X.fit_transform(X_train)

X_test = scale_X.transform(X_test)

print (X_train)
print (X_test)

