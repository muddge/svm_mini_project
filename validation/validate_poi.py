#!/usr/bin/python


"""
    starter code for the validation mini-project
    the first step toward building your POI identifier!

    start by loading/formatting the data

    after that, it's not our code anymore--it's yours!
"""

import pickle
import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "r") )

### first element is our labels, any added elements are predictor
### features. Keep this the same for the mini-project, but you'll
### have a different feature list when you do the final project.
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list)
labels, features = targetFeatureSplit(data)



### it's all yours from here forward!  

# Overfit decision tree

from sklearn import tree
clf = tree.DecisionTreeClassifier()
clf = clf.fit(features, labels)
pred = clf.predict(features)

from sklearn.metrics import accuracy_score
acc = accuracy_score(pred, labels)
print("accuracy [overfit]:", acc)
print("no. of features:", len(features[0]))

# Cross validated decsion tree

import numpy as np
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.30,random_state=42)
clf2 = tree.DecisionTreeClassifier()
clf2 = clf2.fit(features_train, labels_train)
pred2 = clf2.predict(features_test)
acc2 = accuracy_score(pred2, labels_test)
print("accuracy [cross-validated]:", acc2)
print("no. of features:", len(features_test[0]))



