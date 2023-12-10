#%% Libraries
import numpy as np 
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import sklearn.tree as tree
import matplotlib.pyplot as plt
#%% Data Import
my_data = pd.read_csv("drug200.csv", delimiter=",")
my_data.head()
# %% Pre-processing
X = my_data[['Age','Sex', 'BP', 'Cholesterol', 'Na_to_K']].values
X[0:5]

y = my_data["Drug"]
y[0:5]
# %% Pre-processing convert numeric variables
from sklearn import preprocessing
le_Sex = preprocessing.LabelEncoder()
le_Sex.fit(['F','M'])
X[:,1] = le_Sex.transform(X[:,1])

le_BP = preprocessing.LabelEncoder()
le_BP.fit(['LOW','NORMAL','HIGH'])
X[:,2] = le_BP.transform(X[:,2])

le_Chol = preprocessing.LabelEncoder()
le_Chol.fit(['NORMAL','HIGH'])
X[:,3] = le_Chol.transform(X[:,3])

X[0:5]
# %% Setting up the Decision Tree
from sklearn.model_selection import train_test_split
X_trainset, X_testset, y_trainset, y_testset = train_test_split(X,y, test_size=0.3, random_state=99)

print('Shape of X training set {}'.format(X_trainset.shape),'&',' Size of Y training set {}'.format(y_trainset.shape))
print('Shape of X training set {}'.format(X_testset.shape),'&',' Size of Y training set {}'.format(y_testset.shape))
# %% Modeling
drugTree = DecisionTreeClassifier(criterion='entropy', max_depth=4)

drugTree.fit(X_trainset,y_trainset)
# %% Prediction
predTree = drugTree.predict(X_testset)
print (predTree [0:5])
print (y_testset [0:5])

#%% Evaluation
from sklearn import metrics

print("DecisionTrees's Accuracy: ", metrics.accuracy_score(y_testset, predTree))

# %% Visualization

tree.plot_tree(drugTree)
plt.show()
