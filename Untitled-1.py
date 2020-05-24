import numpy as np
import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'
import pydotplus
from IPython.display import Image
from sklearn import tree
from io import StringIO
from graphviz import Source
from IPython.display import display
import graphviz
from sklearn import tree
import pandas as pd
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
df=pd.read_csv('testset.csv')
def plot_decision_tree(clf,feature_name,target_name):
    dot_data = StringIO()
    tree.export_graphviz(clf, out_file=dot_data,
                         feature_names=feature_name,
                         class_names=target_name,
                         filled=True, rounded=True,
                         special_characters=True)
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    return Image(graph.create_png())
del df['datetime_utc']
df['Label'] = pd.factorize(df[' _conds'])[0] + 1
#split dataset in features and target variable

feature_cols = [' _dewptm', ' _fog', ' _hum',' _pressurem']
X = df[feature_cols] # Features
y = df['Label'] # Target variable

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.01, random_state=1) # 70% training and 30% test
# Create Decision Tree classifer object
clf = DecisionTreeClassifier()

# Train Decision Tree Classifer
clf = clf.fit(X_train,y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)


tree.plot_tree(clf)

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))




from IPython.display import Image
from sklearn.externals.six import StringIO
from sklearn.tree import export_graphviz
import pydot

features = list(df.columns[2:10])
print(features)
dotfile = StringIO()
tree.export_graphviz(clf, out_file=dotfile)
graph=pydotplus.graph_from_dot_data(dotfile.getvalue())
graph.write_png("Ashtree.png")

