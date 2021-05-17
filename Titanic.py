# Author: Konstantinos Gyftodimos

# Libraries
import pandas as pd 
import numpy as np
from IPython.display import display
import sklearn
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn import tree
import matplotlib.pyplot as plt
import plotly.express as px
from pydotplus import graph_from_dot_data
from IPython.display import Image

# Train and Test Data
df_train = pd.read_csv('dataset/train.csv')
df_test = pd.read_csv('dataset/test.csv')

# Concatenate the train and test data to do data preparation and put a column indicator to be able to seperate them again in the end:
df = pd.concat([df_test.assign(ind="test"), df_train.assign(ind="train")])
#test, train = df[df["ind"].eq("test")], df[df["ind"].eq("train")]

# Data Preparation and overview of values:
# df_ = df.select_dtypes(exclude=['int', 'float'])
# for col in df_.columns:
#     print(df_[col].unique()) 
#     print(df_[col].value_counts()) 

# 1. Swap the output class: "Survived" with "Embarked"

# 2. Drop Labels that do not matter for the classification like: Passenger Id, Name, Ticket Number
df = df.drop(columns = "Name")
df = df.drop(columns = "PassengerId")
df = df.drop(columns = "Ticket")

# 3. Find possible missing values, errors, duplicate values in feature-columns:

#SURVIVED
#No missing values in Survived output
Survived = np.unique(df['Survived'])


#SEX
#check for mispells or nan values"
df['Sex'] = df['Sex'].replace(['male'],1)
df['Sex'] = df['Sex'].replace(['female'],2)
sex_categories = np.unique(df['Sex'])
# print(sex_categories)
print("Sex Feature: No missing or mispelled values")

#PCLASS
#No missing values in Pclass - feature
Pclass = np.unique(df['Pclass'])
print("Pclass Feature: No missing or mispelled values")

#AGE
#A lot of missing values in Age - feature. Replace NaN values with mean of ages.
Age = np.unique(df['Age']) 
is_age_missing = (df['Age'] == 'NaN')
b = np.logical_not(is_age_missing)
c = df[b]
d = c["Age"]
e = d.astype("float")
is_age_missing_mean = e.mean()
df['Age'].fillna(value=is_age_missing_mean, inplace=True)

#SIBSP
#No missing values in SibSp - feature
SibSp = np.unique(df['SibSp'])

#PARCH
#No missing values in Parch - feature
Parch = np.unique(df['Parch'])

#FARE
#Missing values in Fare - feature
Fare = np.unique(df['Fare'])
count_nan_values_fare = df["Fare"].isna().sum()
print("NaN values in Fares: ",count_nan_values_fare,". Replacing missing value with mean:") 
is_fares_missing = (df['Fare'] == 'NaN')
b = np.logical_not(is_fares_missing)
c = df[b]
d = c["Fare"]
e = d.astype("float")
is_age_missing_mean = e.mean()
df['Fare'].fillna(value=is_fares_missing, inplace=True)
count_nan_values_fare = df["Fare"].isna().sum()
print("NaN values in Fares: ",count_nan_values_fare,". Done!") 

#EMBRARKED
#2 missing categorical values in Embarked - feature. 
count_nan_values_Embarked = df["Embarked"].isna().sum()
print("Number of NaN values in Embarked - feature: ",count_nan_values_Embarked)
#removing rows with nan values from Embarked-feature:
df['Embarked'].fillna(1, inplace=True)

# Correspond values Q,S,C to 1,2,3 respectivelly.
df['Embarked'] = df['Embarked'].replace(['Q'],1)
df['Embarked'] = df['Embarked'].replace(['S'],2)
df['Embarked'] = df['Embarked'].replace(['C'],3)

#CABIN
count_nan_values_cabin = df["Cabin"].isna().sum()
print("Number of NaN values in Cabin - feature: ",count_nan_values_cabin,"... drop this feature :)")
#there are a lot of empty indexes for cabin number so even if i input artificial values and map them to a scalar value, i will not have a representative feature.
#i decide to remove the column.
df = df.drop(columns = "Cabin")


# 4. Data Visualization - Due to the number of features (>2), i will use scatter plots to visualize the data
data_dimensions = df.columns[:-1].to_list()
figure_size = df.shape[1] * 256
fig = px.scatter_matrix(df, dimensions=data_dimensions, color='Survived', width=figure_size, height=figure_size)
#fig.show()


# 5. Re-split the set into train/test:

test, train = df[df["ind"].eq("test")], df[df["ind"].eq("train")]
# remove -ind column from train and test datasets:
train = train.drop(columns = "ind")
test = test.drop(columns = "ind")
# Seperate train and test datasets from their labels:
x_train = train.drop('Survived', axis=1).to_numpy()
y_train = train['Survived'].to_numpy()
x_test = test.drop('Survived', axis=1).to_numpy()
y_test = test['Survived'].to_numpy()


# 6. Reduce dimensionality to 2 features, since most of the features are highly correlated as observed by the previous visualization:

sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

pca = PCA(n_components=2)
x_train = pca.fit_transform(x_train)
x_test = pca.transform(x_test)

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)



# 7. Visualize the reduced features with corresponding classes:
x_train_y_train = np.column_stack((x_train, y_train))
df = pd.DataFrame(x_train_y_train, columns = ["Feature 1","Feature 2","Label"])
fig = px.scatter(df, x="Feature 1", y="Feature 2", color="Label")
#fig.show()



# 8. Binary Classification with a Decision Tree:
classifier = tree.DecisionTreeClassifier(max_leaf_nodes = None, min_samples_leaf = 1, max_depth = 3)
classifier.fit(x_train,y_train)
y_pred = classifier.predict(x_test)

# 9. Create .csv file:
df = pd.DataFrame(y_pred, columns= ['Survived'])
df = df.reset_index()
df = df.rename(columns = {'index': 'PassengerId'}, inplace = False)



df.to_csv('dataset/submission.csv', index = False, header = False)
print(df)