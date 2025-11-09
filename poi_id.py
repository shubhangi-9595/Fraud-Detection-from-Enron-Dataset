#!/usr/bin/python

import sys
import pickle
import pandas as pd
import matplotlib.pyplot
sys.path.append("../tools/")

from tester import test_classifier, dump_classifier_and_data
from feature_format import featureFormat, targetFeatureSplit

import numpy as np
from time import time
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
from sklearn.cross_validation import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score, precision_score, recall_score

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
financial_features = ['salary', 'deferral_payments', 'total_payments',
                      'loan_advances', 'bonus', 'restricted_stock_deferred',
                      'deferred_income', 'total_stock_value', 'expenses',
                      'exercised_stock_options', 'other', 'long_term_incentive',
                      'restricted_stock', 'director_fees']
email_features =  ['to_messages', 'from_poi_to_this_person',
                   'from_messages', 'from_this_person_to_poi', 'shared_receipt_with_poi'] 
poi_label = ['poi']
features_list = poi_label + email_features + financial_features

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)
    
#EXPLORING DATA
#Total number of people
number_of_people = len(data_dict)
print("Total number of people: %i" %number_of_people)
#Number of POI and NON-POI
poi = 0
for person in data_dict:
    if data_dict[person]['poi']:
       poi = poi + 1
print("Total number of poi: %i" % poi)
print("Total number of non-poi: %i" % (number_of_people - poi))
#Count of Features
all_features = data_dict[data_dict.keys()[0]].keys()
print("There are %i features for each person in the dataset, and %i features are used" %(len(all_features), len(features_list)))

#Missing Values
missing_values = {}
#Get all features list
features = data_dict[data_dict.keys()[0]].keys()
#Assign missing value for all features to 0
for feature in features:
    missing_values[feature] = 0
#Loop through each person &
#increase count of missing_values for each empty feature
for person in data_dict:
    for feature in data_dict[person]:
        if data_dict[person][feature] == "NaN":
            missing_values[feature] = missing_values[feature] + 1
print("Number of missing values for each feature are:")
for feature in missing_values:
    print feature, missing_values[feature]

### Task 2: Remove outliers
#STEP 1: IDENTIFY OUTLIERS
#Draw scatter plot
def generateScatterPlot(data_set, feature_x, feature_y):
    data = featureFormat(data_set, [feature_x, feature_y])
    for point in data:
        x = point[0]
        y = point[1]
        matplotlib.pyplot.scatter( x, y )
    matplotlib.pyplot.xlabel(feature_x)
    matplotlib.pyplot.ylabel(feature_y)
    matplotlib.pyplot.show()
#Identify outliers
generateScatterPlot(data_dict, 'total_payments', 'total_stock_value')
generateScatterPlot(data_dict, 'from_poi_to_this_person', 'from_this_person_to_poi')
generateScatterPlot(data_dict, 'salary', 'bonus')
generateScatterPlot(data_dict, 'total_payments', 'other')
#STEP 2: REMOVE OUTLIERS
print "Names of people in the dataset: "
s = []
for person in data_dict.keys():
    s.append(person)
    if len(s) == 4:
        print '{:<30}{:<30}{:<30}{:<30}'.format(s[0],s[1],s[2],s[3])
        s = []
print '{:<30}{:<30}'.format(s[0],s[1])
#TOTAL is an outlier
data_dict.pop('TOTAL', 0)

### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
my_dataset = data_dict

#CREATE MESSAGE RATIO FEATURES - from_poi_message_ratio & to_poi_message_ratio
for person in my_dataset.values():
    person['to_poi_message_ratio'] = 0
    person['from_poi_message_ratio'] = 0
    if float(person['to_messages']) > 0:
        person['to_poi_message_ratio'] = float(person['from_this_person_to_poi'])/float(person['to_messages'])
    if float(person['from_messages']) > 0:        
        person['from_poi_message_ratio'] = float(person['from_poi_to_this_person'])/float(person['from_messages'])
features_list.extend(['to_poi_message_ratio', 'from_poi_message_ratio'])

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

#Select the best features: 
#Removes features whose variance is below 80%
from sklearn.feature_selection import VarianceThreshold
sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
features = sel.fit_transform(features)
#Removes all but the k highest scoring features
k=7
selector = SelectKBest(f_classif, k=7)
selector.fit_transform(features, labels)
scores = zip(features_list[1:],selector.scores_)
sorted_scores = sorted(scores, key = lambda x: x[1], reverse=True)
print sorted_scores
optimized_features_list = poi_label + list(map(lambda x: x[0], sorted_scores))[0:k]
print "Best features are:"
print optimized_features_list

# Extract from dataset without new features
data = featureFormat(my_dataset, optimized_features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)
scaler = preprocessing.MinMaxScaler()
features = scaler.fit_transform(features)
# Extract from dataset with new features
data = featureFormat(my_dataset, optimized_features_list + \
                     ['to_poi_message_ratio', 'from_poi_message_ratio'], \
                     sort_keys = True)
new_f_labels, new_f_features = targetFeatureSplit(data)
new_f_features = scaler.fit_transform(new_f_features)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html
def evaluate(grid_search, features, labels, params, iters=100):
    precision = []
    accuracy = []
    recall = []
    for i in range(iters):
        features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.3, random_state=i)
        grid_search.fit(features_train, labels_train)
        predictions = grid_search.predict(features_test)
        precision = precision + [precision_score(labels_test, predictions)]
        accuracy = accuracy + [accuracy_score(labels_test, predictions)] 
        recall = recall + [recall_score(labels_test, predictions)]
    print "Precision =", np.mean(precision)
    print "Accuracy =", np.mean(accuracy)
    print "Recall =", np.mean(recall)
    best_params = grid_search.best_estimator_.get_params()
    for param_name in params.keys():
        print("%s = %r, " % (param_name, best_params[param_name]))

#Naive Bayesian
from sklearn.naive_bayes import GaussianNB       
clf = GaussianNB()
param = {}
grid_search = GridSearchCV(clf, param)
print("Naive Bayesian Model Results:")
evaluate(grid_search, features, labels, param)
'''
RESULT
Precision = 0.432977633478
Accuracy = 0.854761904762
Recall = 0.373191558442
'''

#SVM
from sklearn import svm
svm_clf = svm.SVC()
svm_param = {'kernel': ['rbf', 'linear', 'poly'], 'C': [0.1, 1, 10, 100, 1000],\
           'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 'random_state': [42]}
svm_grid_search = GridSearchCV(svm_clf, svm_param)
print("SVM Results:")
evaluate(svm_grid_search, features, labels, svm_param)
'''
RESULT
Precision = 0.141666666667
Accuracy = 0.866428571429
Recall = 0.0384523809524
kernel = 'linear', 
C = 1, 
random_state = 42, 
gamma = 1,
'''

#Decison Tree
from sklearn import tree
dt_clf = tree.DecisionTreeClassifier()
dt_param = {'criterion':['gini','entropy'],'splitter':['best','random'], 'random_state': [0]}
dt_grid_search = GridSearchCV(dt_clf, dt_param)
print("Decison Tree Results:")
evaluate(dt_grid_search, features, labels, dt_param)
'''
RESULT
Precision = 0.209663695781
Accuracy = 0.79380952381
Recall = 0.242603535354
splitter = 'best', 
random_state = 0, 
criterion = 'gini',
'''

#RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
rf1_clf = RandomForestClassifier()
rf1_param = {'criterion':['gini','entropy'], 'random_state': [20]}
rf1_grid_search = GridSearchCV(rf1_clf, rf1_param)
print("Random Forest Classifier Results:")
evaluate(rf1_grid_search, features, labels, rf1_param)
'''
RESULT
Precision = 0.368952380952
Accuracy = 0.858095238095
Recall = 0.140107503608
random_state = 20, 
criterion = 'gini',
'''

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

#Naive Bayesian
from sklearn.naive_bayes import GaussianNB       
clf = GaussianNB()
param = {}
grid_search = GridSearchCV(clf, param)
print("Naive Bayesian Model Results:")
evaluate(grid_search, features, labels, param)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, optimized_features_list)
