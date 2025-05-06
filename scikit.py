# MACHINE LEARNING

import sklearn
from sklearn.datasets import load_iris
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score

                                       # datasets
X, y = load_iris(return_X_y =True)
# X1, y1 = load_breast_cancer(return_X1_y1 =True)

                                        # splitting data sets BASED ON YOUR PREFRENCES
xtrain , xtest , ytrain , ytest =train_test_split (X,y, train_size=0.6)
# Xtrain , Xtest , Ytrain , Ytest = train_test_split(X1,y1 , train_size=0.8)


                                         #  classifiers

dt= DecisionTreeClassifier()
rf = RandomForestClassifier()
nb = GaussianNB()
knn =KNeighborsClassifier()
lrn = LogisticRegression()
sv = svm.SVC()
  
                                              # print
print(X.shape)
print(y.shape)
print(xtrain.shape)
print(xtest.shape)
print(ytrain.shape)
print(ytest.shape)

                   
#  training phase
# dt.fit(xtrain,ytrain)

                                     
# testing phase
# acc= dt.score(xtest ,ytest)
# print(round(acc *100),2)


                  #to check the accuracy of the data sets we have to predict first 

# dt_acc = accuracy_score(dt.predict(xtest), ytest)
# print(dt_acc)
                  # upto 2 decimal points
# accScore =accuracy_score(dt.predict(xtest) , ytest)
# print(round(accScore),2)



                  #  ***************CROSS VALIDATION***************

             # USING DECESION TREE 
# cv_accuracy = cross_val_score(dt,X,y,cv=10)
# print(cv_accuracy.mean())
# print(cv_accuracy.std())

            #USING RANDOMFOREST CLASSIFIER

                                 #using 5 cv 
# cv_accuracy  = cross_val_score(rf,X,y,cv=5)  
# print(cv_accuracy.mean())
# print(cv_accuracy.std()) 
                                 #using 10 cv 
# cv_accuracy_one  = cross_val_score(rf,X,y,cv=10)  
# print(cv_accuracy_one.mean())
# print(cv_accuracy_one.std())
 
  
            #USING GUASSAIN NB

                               #using 5 cv
# cv_accuracy  = cross_val_score(nb,X,y,cv=5)  
# print(cv_accuracy.mean())
# print(cv_accuracy.std())  
                              #using 10 cv                              
# cv_accuracy_one  = cross_val_score(nb,X,y,cv=10)  
# print(cv_accuracy_one.mean())
# print(cv_accuracy_one.std()) 


            #USING KNeighborsClassifier

                              #using 5 cv
# cv_accuracy  = cross_val_score(knn,X,y,cv=5)  
# print(cv_accuracy.mean())
# print(cv_accuracy.std()) 
                              #using 10 cv                              
# cv_accuracy_one  = cross_val_score(knn,X,y,cv=10)  
# print(cv_accuracy_one.mean())
# print(cv_accuracy_one.std()) 


            #USING LOGISTIC REGRESSION

                              #using 5 cv
# cv_accuracy  = cross_val_score(lrn,X,y,cv=5)  
# print(cv_accuracy.mean())
# print(cv_accuracy.std()) 
                              #using 10 cv                              
# cv_accuracy_one  = cross_val_score(lrn,X,y,cv=10)  
# print(cv_accuracy_one.mean())
# print(cv_accuracy_one.std()) 


            #USING SV.SVM

                              #using 5 cv
# cv_accuracy  = cross_val_score(sv,X,y,cv=5)  
# print(cv_accuracy.mean())
# print(round(cv_accuracy.mean(),2))
# print(cv_accuracy.std()) 
# print(round(cv_accuracy.std(),2))
                              #using 10 cv                              
# cv_accuracy_one  = cross_val_score(sv,X,y,cv=10)  
# print(cv_accuracy_one.mean())
# print(cv_accuracy_one.std()) 




