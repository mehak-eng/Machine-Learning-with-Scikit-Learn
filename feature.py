
# BIG DATA ANALYTICS

# load scoring techniques on the basis of features are ranked


# load feature selection methods 
from sklearn.feature_selection import  SelectKBest , SelectPercentile ,SelectFdr , SelectFpr , SelectFwe , GenericUnivariateSelect ,f_classif, mutual_info_classif , chi2 

# load dataset
from sklearn.datasets import load_breast_cancer
X,y = load_breast_cancer(return_X_y=True)
print(X.shape)

# use the selection models with scoring techniques

# # SELECT K-BEST
k_chi = SelectKBest(chi2 , k=5).fit_transform(X,y)
print(k_chi.shape)

k_mut = SelectKBest(mutual_info_classif, k=5).fit_transform(X,y)
print(k_mut.shape)

k_classif =SelectKBest(f_classif , k=3).fit_transform(X,y)
print(k_classif)

# # SELECT PERCENTILE 
per_chi = SelectPercentile(chi2 , percentile=15).fit_transform(X,y)
print(per_chi.shape)

per_mut =SelectPercentile(mutual_info_classif , percentile=5).fit_transform(X,y)
print(per_mut.shape)

per_classif =SelectPercentile(f_classif , percentile=5).fit_transform(X,y)
print(per_classif.shape)

# # SELECT FDR (false discovery rate)

dr_chi = SelectFdr(chi2 , alpha=0.2).fit_transform(X,y)
print(dr_chi.shape)

dr_classif = SelectFdr(f_classif , alpha=0.7).fit_transform(X,y)
print(dr_classif.shape)


# WE CANNOT USE FDR SELECTION METHOD WITH MUTUAL_CLASSIF SCORING TEXHNIQUE BECAUSE IT DOESNOT SUPOORT IT


# # SELECT FPR (false positive rate)

pr_chi = SelectFpr(chi2 , alpha=0.3).fit_transform(X,y)
print(pr_chi.shape)

pr_classif = SelectFpr(f_classif , alpha=0.006).fit_transform(X,y)
print(pr_classif.shape)


# WE CANNOT USE FPR SELECTION METHOD WITH MUTUAL_CLASSIF SCORING TEXHNIQUE BECAUSE IT DOESNOT SUPOORT IT


# # SELECT FWE(family wise error)

we_chi = SelectFwe(chi2 , alpha=0.03).fit_transform(X,y)
print(we_chi.shape)

we_classif = SelectFwe(f_classif , alpha=0.03).fit_transform(X,y)
print(we_classif.shape)


# WE CANNOT USE FWE SELECTION METHOD WITH MUTUAL_CLASSIF SCORING TEXHNIQUE BECAUSE IT DOESNOT SUPOORT IT

print("*************************************************************")

# TO GET THE NAMES OF FEATURES SELECTED 

data = load_breast_cancer()

# X is the data  y is the labels/target  

X, y = data.data, data.target
feature_names = data.feature_names

# Generic Univarient Method

transformer = GenericUnivariateSelect(chi2, mode='k_best', param=20)
X_new = transformer.fit_transform(X, y)
print(X_new.shape)

# # Select K_best method

selector = SelectKBest(score_func=chi2, k=2)
X_selected = selector.fit_transform(X, y)
# Get feature names
selected_features = feature_names[selector.get_support()]
print("Selected feature names:")
print(selected_features)


# #Select Percentile method

per_mu = SelectPercentile(mutual_info_classif , percentile=2).fit(X , y)

# fit func is used to fit the dataset along with arrays 

# Get feature names
trans_mu = per_mu.transform(X)

# transform is used to tranformed the fit data 
# while fit_transform done parallel

print(feature_names[per_mu.get_support()])
print(trans_mu)



#select fpr method
pr_chi = SelectFpr(chi2 ,alpha = 0.05).fit(X , y)
# Get feature names
trans_chi = pr_chi.transform(X)
print(feature_names[pr_chi.get_support()])
print(trans_chi)



