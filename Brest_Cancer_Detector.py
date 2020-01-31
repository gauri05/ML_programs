import numpy as np
from sklearn.datasets import load_breast_cancer
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
import time
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC,NuSVC,LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier

cancer=pd.read_csv('breast_cancer_dataset.csv')
print("cancer.keys(): \n{}".format(cancer.keys()))

cancer.info()
cancer.head(3)

#print("Shape of cancer data: {}".format(cancer.data.shape))

#print("sample counts per class:\n{}".format({n: v for n, v in zip(cancer.target_names, np.bincount(cancer.target))}))

#print("Feature names:\n{}".format(cancer.feature_names))

#print(cancer.head(3))

cancer.drop(cancer.columns[[-1,0]],axis=1,inplace=True)
cancer.info()

diagnosis_all=list(cancer.shape)[0]
diagnosis_categories=list(cancer['diagnosis'].value_counts())

print("\n \t The data has {} diagnosis, {} malignant and {} benign.".format(diagnosis_all,diagnosis_categories[0],diagnosis_categories[1]))

features_mean=list(cancer.columns[1:11])

#The red dots correspond to malignant diagnosis and blue to benign. Look how in some cases reds and blues dots occupies different regions of the plots.

plt.figure(figsize=(15,15))
sns.heatmap(cancer[features_mean].corr(),annot=True,square=True,cmap='coolwarm')
plt.show()

color_dic={'M':'red','B':'blue'}
colors=cancer['diagnosis'].map(lambda x:color_dic.get(x))

sm=pd.plotting.scatter_matrix(cancer[features_mean],c=colors,alpha=0.4,figsize=((15,15)))

plt.show()

#We can also see how the malignant or benign tumors cells can have (or not) different values for the features plotting the distribution of each type of diagnosis for each of the mean features.
bins=12
plt.figure(figsize=(15,15))

for i, feature in enumerate(features_mean):
    rows=int(len(features_mean)/2)

    plt.subplot(rows,2,i+1)

    sns.distplot(cancer[cancer['diagnosis']=='M'][feature],bins=bins,color='red',label='M')
    sns.distplot(cancer[cancer['diagnosis'] == 'B'][feature], bins=bins, color='blue', label='B')

    plt.legend(loc='upper right')

plt.tight_layout()
plt.show()

#Still another form of doing this could be using box plots, which is done below.

plt.figure(figsize=(15,15))
for i, feature in enumerate(features_mean):
    rows=int(len(features_mean)/2)

    plt.subplot(rows,2,i+1)

    sns.boxplot(x='diagnosis',y=feature,data=cancer,palette="Set1")

plt.tight_layout()
plt.show()

features_selection = ['radius_mean', 'perimeter_mean', 'area_mean', 'concavity_mean', 'concave points_mean']

diag_map={'M':1,'B':0}
cancer['diagnosis']=cancer['diagnosis'].map(diag_map)

X=cancer.loc[:,features_mean]
y=cancer.loc[:,'diagnosis']

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

accuracy_all=[]
cvs_all=[]


# Stochastic Gradient Descent
start=time.time()

clf=SGDClassifier()
clf.fit(X_train,y_train)
prediction=clf.predict(X_test)
score=cross_val_score(clf,X,y,cv=5)

end=time.time()

accuracy_all.append(accuracy_score(prediction,y_test))
cvs_all.append(np.mean(score))

print("SGD Classifier Accuracy: {0:.2%}".format(accuracy_score(prediction,y_test)))
print("Cross validation score:{0:.2%} (+/- {1:.2%})".format(np.mean(score),np.std(score)*2))
print("Execution time: {0:.5} seconds \n".format(end-start))

start=time.time()

clf=SVC()
clf.fit(X_train,y_train)
prediction=clf.predict(X_test)
score=cross_val_score(clf,X,y,cv=5)

end=time.time()

accuracy_all.append(accuracy_score(prediction,y_test))
cvs_all.append(np.mean(score))

print("SVC Accuracy: {0:.2%}".format(accuracy_score(prediction,y_test)))
print("Cross validation score: {0:.2%} (+/- {1:.2%})".format(np.mean(score), np.std(score)*2))
print("Execution time: %s seconds \n" % "{0:.5}".format(end-start))

start=time.time()

clf=NuSVC()
clf.fit(X_train,y_train)
prediction=clf.predict(X_test)
score=cross_val_score(clf,X,y,cv=5)

end=time.time()

accuracy_all.append(accuracy_score(prediction,y_test))
cvs_all.append(np.mean(score))

print("NuSVC Accuracy: {0:.2%}".format(accuracy_score(prediction,y_test)))
print("Cross validation score: {0:.2%} (+/- {1:.2%})".format(np.mean(score),np.std(score)*2))
print("Execution time: %s seconds \n" % "{0:.5}".format(end-start))

start=time.time()

clf=LinearSVC()
clf.fit(X_train,y_train)
prediction=clf.predict(X_test)
score=cross_val_score(clf,X,y,cv=5)

end=time.time()

accuracy_all.append(accuracy_score(prediction,y_test))
cvs_all.append(np.mean(score))

print("LinearSVC Accuracy: {0:.2%}".format(accuracy_score(prediction,y_test)))
print("Cross validation score: {0:.2%} (+/- {1:.2%})".format(np.mean(score),np.std(score)*2))
print("Execution time: %s seconds \n" % "{0:.5}".format(end-start))

start=time.time()
clf=KNeighborsClassifier()
clf.fit(X_train,y_train)
prediction=clf.predict(X_test)
score=cross_val_score(clf,X,y,cv=5)

end=time.time()

accuracy_all.append(accuracy_score(prediction,y_test))
cvs_all.append(np.mean(score))

print("Accuracy: {0:.2%}".format(accuracy_score(prediction,y_test)))
print("Cross validation score: {0:.2%} (+/- {1:.2%})".format(np.mean(score),np.std(score)*2))
print("Execution time: %s seconds \n" % "{0:.5}".format(end-start))

# Naive Bayes
start=time.time()

clf=GaussianNB()
clf.fit(X_train,y_train)
prediction=clf.predict(X_test)
score=cross_val_score(clf,X,y,cv=5)

end=time.time()

accuracy_all.append(accuracy_score(prediction,y_test))
cvs_all.append(np.mean(score))

print("Accuracy: {0:.2%}".format(accuracy_score(prediction,y_test)))
print("Cross validation score: {0:.2%} (+/- {1:.2%})".format(np.mean(score),np.std(score)*2))
print("Execution time: %s seconds \n" % "{0:.5}".format(end-start))

# Forest and tree methods

start=time.time()
clf=RandomForestClassifier()
clf.fit(X_train,y_train)
prediction=clf.predict(X_test)
score=cross_val_score(clf,X,y,cv=5)

end=time.time()

accuracy_all.append(accuracy_score(prediction,y_test))
cvs_all.append(np.mean(score))

print("Random Forest Accuracy: {0:.2%}".format(accuracy_score(prediction,y_test)))
print("Cross validation score: {0:.2%} (+/- {1:.2%})".format(np.mean(score),np.std(score)*2))
print("Execution time: %s seconds \n" % "{0:.5}".format(end-start))

clf = ExtraTreesClassifier()
clf.fit(X_train, y_train)
prediction = clf.predict(X_test)
scores = cross_val_score(clf, X, y, cv=5)

end = time.time()

accuracy_all.append(accuracy_score(prediction, y_test))
cvs_all.append(np.mean(scores))

print("Extra Trees Accuracy: {0:.2%}".format(accuracy_score(prediction, y_test)))
print("Cross validation score: {0:.2%} (+/- {1:.2%})".format(np.mean(scores), np.std(scores)*2))
print("Execution time: %s seconds \n" % "{0:.5}".format(end-start))

start = time.time()

clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)
prediction = clf.predict(X_test)
scores = cross_val_score(clf, X, y, cv=5)

end = time.time()

accuracy_all.append(accuracy_score(prediction, y_test))
cvs_all.append(np.mean(scores))

print("Dedicion Tree Accuracy: {0:.2%}".format(accuracy_score(prediction, y_test)))
print("Cross validation score: {0:.2%} (+/- {1:.2%})".format(np.mean(scores), np.std(scores)*2))
print("Execution time: %s seconds \n" % "{0:.5}".format(end-start))


######## feature_selected

X=cancer.loc[:,features_selection]
y=cancer.loc[:,'diagnosis']

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

accuracy_selection = []
cvs_selection = []

# Stochastic Gradient Descent
start=time.time()

clf=SGDClassifier()
clf.fit(X_train,y_train)
prediction=clf.predict(X_test)
score=cross_val_score(clf,X,y,cv=5)

end=time.time()

accuracy_selection.append(accuracy_score(prediction,y_test))
cvs_selection.append(np.mean(score))

print("SGD Classifier Accuracy: {0:.2%}".format(accuracy_score(prediction,y_test)))
print("Cross validation score:{0:.2%} (+/- {1:.2%})".format(np.mean(score),np.std(score)*2))
print("Execution time: {0:.5} seconds \n".format(end-start))

start=time.time()

clf=SVC()
clf.fit(X_train,y_train)
prediction=clf.predict(X_test)
score=cross_val_score(clf,X,y,cv=5)

end=time.time()

accuracy_selection.append(accuracy_score(prediction,y_test))
cvs_selection.append(np.mean(score))

print("SVC Accuracy: {0:.2%}".format(accuracy_score(prediction,y_test)))
print("Cross validation score: {0:.2%} (+/- {1:.2%})".format(np.mean(score), np.std(score)*2))
print("Execution time: %s seconds \n" % "{0:.5}".format(end-start))

start=time.time()

clf=NuSVC()
clf.fit(X_train,y_train)
prediction=clf.predict(X_test)
score=cross_val_score(clf,X,y,cv=5)

end=time.time()

accuracy_selection.append(accuracy_score(prediction,y_test))
cvs_selection.append(np.mean(score))

print("NuSVC Accuracy: {0:.2%}".format(accuracy_score(prediction,y_test)))
print("Cross validation score: {0:.2%} (+/- {1:.2%})".format(np.mean(score),np.std(score)*2))
print("Execution time: %s seconds \n" % "{0:.5}".format(end-start))

start=time.time()

clf=LinearSVC()
clf.fit(X_train,y_train)
prediction=clf.predict(X_test)
score=cross_val_score(clf,X,y,cv=5)

end=time.time()

accuracy_selection.append(accuracy_score(prediction,y_test))
cvs_selection.append(np.mean(score))

print("LinearSVC Accuracy: {0:.2%}".format(accuracy_score(prediction,y_test)))
print("Cross validation score: {0:.2%} (+/- {1:.2%})".format(np.mean(score),np.std(score)*2))
print("Execution time: %s seconds \n" % "{0:.5}".format(end-start))

start=time.time()
clf=KNeighborsClassifier()
clf.fit(X_train,y_train)
prediction=clf.predict(X_test)
score=cross_val_score(clf,X,y,cv=5)

end=time.time()

accuracy_selection.append(accuracy_score(prediction,y_test))
cvs_selection.append(np.mean(score))

print("Accuracy: {0:.2%}".format(accuracy_score(prediction,y_test)))
print("Cross validation score: {0:.2%} (+/- {1:.2%})".format(np.mean(score),np.std(score)*2))
print("Execution time: %s seconds \n" % "{0:.5}".format(end-start))

# Naive Bayes
start=time.time()

clf=GaussianNB()
clf.fit(X_train,y_train)
prediction=clf.predict(X_test)
score=cross_val_score(clf,X,y,cv=5)

end=time.time()

accuracy_selection.append(accuracy_score(prediction,y_test))
cvs_selection.append(np.mean(score))

print("Accuracy: {0:.2%}".format(accuracy_score(prediction,y_test)))
print("Cross validation score: {0:.2%} (+/- {1:.2%})".format(np.mean(score),np.std(score)*2))
print("Execution time: %s seconds \n" % "{0:.5}".format(end-start))

# Forest and tree methods

start=time.time()
clf=RandomForestClassifier()
clf.fit(X_train,y_train)
prediction=clf.predict(X_test)
score=cross_val_score(clf,X,y,cv=5)

end=time.time()

accuracy_selection.append(accuracy_score(prediction,y_test))
cvs_selection.append(np.mean(score))

print("Random Forest Accuracy: {0:.2%}".format(accuracy_score(prediction,y_test)))
print("Cross validation score: {0:.2%} (+/- {1:.2%})".format(np.mean(score),np.std(score)*2))
print("Execution time: %s seconds \n" % "{0:.5}".format(end-start))

clf = ExtraTreesClassifier()
clf.fit(X_train, y_train)
prediction = clf.predict(X_test)
scores = cross_val_score(clf, X, y, cv=5)

end = time.time()

accuracy_selection.append(accuracy_score(prediction, y_test))
cvs_selection.append(np.mean(scores))

print("Extra Trees Accuracy: {0:.2%}".format(accuracy_score(prediction, y_test)))
print("Cross validation score: {0:.2%} (+/- {1:.2%})".format(np.mean(scores), np.std(scores)*2))
print("Execution time: %s seconds \n" % "{0:.5}".format(end-start))

start = time.time()

clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)
prediction = clf.predict(X_test)
scores = cross_val_score(clf, X, y, cv=5)

end = time.time()

accuracy_selection.append(accuracy_score(prediction, y_test))
cvs_selection.append(np.mean(scores))

print("Dedicion Tree Accuracy: {0:.2%}".format(accuracy_score(prediction, y_test)))
print("Cross validation score: {0:.2%} (+/- {1:.2%})".format(np.mean(scores), np.std(scores)*2))
print("Execution time: %s seconds \n" % "{0:.5}".format(end-start))











diff_accuracy = list(np.array(accuracy_selection) - np.array(accuracy_all))
diff_cvs = list(np.array(cvs_selection) - np.array(cvs_all))

d = {'accuracy_all':accuracy_all, 'accuracy_selection':accuracy_selection, 'diff_accuracy':diff_accuracy,
     'cvs_all':cvs_all, 'cvs_selection':cvs_all, 'diff_cvs':diff_cvs,}

index = ['SGD', 'SVC', 'NuSVC', 'LinearSVC', 'KNeighbors', 'GaussianNB', 'RandomForest', 'ExtraTrees', 'DecisionTree']

df = pd.DataFrame(d, index=index)

print(df)


#Y=cancer['diagnosis']
#Y = cancer['diagnosis']
#X = cancer.drop(["diagnosis"], axis = 1, inplace = True)
#print(X)


# print(Y)
#
# print("Cancer data set dimensions : {}".format(cancer.shape))

# Encoding categorical data values
# from sklearn.preprocessing import LabelEncoder
# labelencoder_y=LabelEncoder()
#Y=labelencoder_y.fit_transform(Y)

#print(Y)



# Using Logistic Regression Algorithm to the training set
# from sklearn.linear_model import LogisticRegression
# classifier=LogisticRegression(random_state=0)
#classifier.fit()