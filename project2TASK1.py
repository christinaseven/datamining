import sklearn
print(sklearn.__version__)
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.model_selection import train_test_split
from numpy import mean
from numpy import std
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.ensemble import BaggingClassifier
dataset = pd.read_csv("Seventikidou.csv", names=['c1', 'c2','c3','c4','c5','c6','c7','c8','c9','c10','c11','c12','c13','target'])
array=dataset.values #Return a Numpy representation of the DataFrame(dataset is a dataframe)
X = array[:,0:13]
Y = array[:,13]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=1)
model = BaggingClassifier()
# evaluate the model
scores = cross_val_score(model, X_train, Y_train, scoring='accuracy', cv=10)
# report performance
print('Accuracy: %.3f (%.3f)' % (mean(scores), std(scores)))
from matplotlib import pyplot
fig = pyplot.figure()

models=[] 
models.append(('25',BaggingClassifier(n_estimators=25)))
models.append(('50',BaggingClassifier(n_estimators=50)))
models.append(('75',BaggingClassifier(n_estimators=75)))
models.append(('100',BaggingClassifier(n_estimators=100)))
names=[]
results=[]
for name, model in models:
    scores = cross_val_score(model, X_train, Y_train, cv=10, scoring='accuracy')
    results.append(scores)
    names.append(name)
    print("Accuracy of ensemble model with number of estimators")
    print('%s: %f (%f)' % (name, scores.mean(), scores.std()))
pyplot.boxplot(results, labels=names, showmeans=True)
pyplot.show()
model = BaggingClassifier(n_estimators=100) #bagging final model
#fit the model on the whole dataset
trainmodel= model.fit(X_train, Y_train)
# prediction
Y_pred = trainmodel.predict(X_test)
    
acc = accuracy_score(Y_test,Y_pred)
print('The accuracy: {}'.format(acc))
#random forest
from sklearn.ensemble import RandomForestClassifier
models=[]
models.append(('25',RandomForestClassifier(n_estimators=25, oob_score=True, min_samples_leaf=5)))
models.append(('50',RandomForestClassifier(n_estimators=50, oob_score=True, min_samples_leaf=5)))
models.append(('75',RandomForestClassifier(n_estimators=75, oob_score=True, min_samples_leaf=5)))
models.append(('100',RandomForestClassifier(n_estimators=100, oob_score=True, min_samples_leaf=5)))
names=[]
results=[]
print("Random forest results: ")
for name, model in models:
    model.fit(X_train,Y_train)
    oob_error = 1 - model.oob_score_
    results.append(oob_error)
    names.append(name)
    print("Error of ensemble model with number of estimators: "+name)
    print(oob_error)
    
model= RandomForestClassifier(n_estimators=100, oob_score=True, min_samples_leaf=5)
model.fit(X_train, Y_train)
#oob_score_ 
oob_score = model.oob_score_
print("out of bag accuracy score is: ") 
print(oob_score)

Y_pred = model.predict(X_test)
print("Accuracy classification score to the test set and the prediction labels: ")
print(accuracy_score(Y_test, Y_pred))
feature_names=['c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9', 'c10', 'c11',
       'c12', 'c13']
target_names=['0','1']
feature_imp = pd.Series(model.feature_importances_,index=feature_names).sort_values(ascending=False)
feature_imp
#visulization of important features
import matplotlib.pyplot as plt
import seaborn as sns
# Creating a bar plot
sns.barplot(x=feature_imp, y=feature_imp.index)
# Add labels to your graph
plt.xlabel('Feature Importance')
plt.ylabel('Features')
plt.title("Visualizing Important Features")
plt.show()
data=dataset.copy()
 
pyplot.tight_layout() 

del data['c3']
del data['c5']
del data['c4']
del data['c7']
array=data.values #Return a Numpy representation of the DataFrame(
x = array[:,0:9]
y = array[:,9]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=1)
m=RandomForestClassifier(n_estimators=100,oob_score=True, min_samples_leaf=5)
m.fit(x_train,y_train)

# prediction on test set
y_pred=m.predict(x_test)
print("Accuracy:",accuracy_score(y_test, y_pred))
print("oob score", m.oob_score_)
