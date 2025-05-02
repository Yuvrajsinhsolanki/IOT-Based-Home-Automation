import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import cross_val_score
import seaborn as sns
from sklearn import tree
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('crop_recommendation.csv')
print(df.head())
print("Shape:-",df.shape)
print("Columns:-",df.columns)
print("Data Types:-",df.dtypes)

features = df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
target = df['crop']

Xtrain, Xtest, Ytrain, Ytest = train_test_split(features,target,test_size = 0.2,random_state =2)

Random_Forest = RandomForestClassifier(n_estimators=20, random_state=5)
Random_Forest.fit(Xtrain,Ytrain)

predicted_values = Random_Forest.predict(Xtest)

acc=[]
model=[]
x = metrics.accuracy_score(Ytest, predicted_values)
acc.append(x)
model.append('Random_Forest')
print("Random Forest Accuracy is: ", x)
print(classification_report(Ytest,predicted_values))

score = cross_val_score(Random_Forest,features,target,cv=5)
print("Cross Validation Score is: ", score.mean())

RandomForest_pkl_filename = 'Random_Forest.pkl'
RandomForest_Model_pkl = open(RandomForest_pkl_filename, 'wb')
pickle.dump(Random_Forest, RandomForest_Model_pkl)
RandomForest_Model_pkl.close()

data = np.array([[104,18, 30, 23.603016, 60.3, 6.7, 140.91]])
prediction = Random_Forest.predict(data)
print(prediction)