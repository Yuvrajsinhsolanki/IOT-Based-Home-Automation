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

# Load the dataset
df = pd.read_csv('Fertilizer_recommendation.csv')
print(df.head())
print("Shape:-", df.shape)
print("Columns:-", df.columns)
print("Data Types:-", df.dtypes)

# Encode categorical columns
label_encoder_soil = LabelEncoder()
label_encoder_crop = LabelEncoder()

df['Soil Type'] = label_encoder_soil.fit_transform(df['Soil Type'])
df['Crop Type'] = label_encoder_crop.fit_transform(df['Crop Type'])

# Define features and target
features = df[['Temperature', 'Humidity', 'Soil Type', 'Crop Type', 'Nitrogen', 'Phosphorus', 'Potassium']]
target = df['Fertilizer Name']

# Train-test split
Xtrain, Xtest, Ytrain, Ytest = train_test_split(features, target, test_size=0.2, random_state=2)

# Train Random Forest model
Random_Forest = RandomForestClassifier(n_estimators=20, random_state=5)
Random_Forest.fit(Xtrain, Ytrain)

# Make predictions
predicted_values = Random_Forest.predict(Xtest)

# Evaluate model
acc = []
model = []
x = metrics.accuracy_score(Ytest, predicted_values)
acc.append(x)
model.append('Random_Forest')
print("Random Forest Accuracy is: ", x)
print(classification_report(Ytest, predicted_values))

# Cross-validation score
score = cross_val_score(Random_Forest, features, target, cv=5)
print("Cross Validation Score is: ", score.mean())

# Save the model
RandomForest_pkl_filename = 'Random_Forest.pkl'
with open(RandomForest_pkl_filename, 'wb') as RandomForest_Model_pkl:
    pickle.dump(Random_Forest, RandomForest_Model_pkl)

# Handle unseen labels for prediction
try:
    new_data = np.array([[24, 34, 
                          label_encoder_soil.transform(['Black'])[0] if 'Black' in label_encoder_soil.classes_ else -1, 
                          label_encoder_crop.transform(['Oil Seed'])[0] if 'Oil Seed' in label_encoder_crop.classes_ else -1, 
                          60.3, 6.7, 15]])
    prediction = Random_Forest.predict(new_data)
    print("Prediction for new data: ", prediction)
except ValueError as e:
    print("Error in prediction due to unseen labels:", e)