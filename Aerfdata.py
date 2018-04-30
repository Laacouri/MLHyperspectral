
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split


# In[2]:


Hyper=pd.read_csv("C:\Users\garag\Google Drive\Hyperspectral\Auto_band.csv")


# In[3]:


Hyper.shape


# In[5]:


## Above, I show the dimension of the labeled data. 10078 sample, 240 bands and one column for the label.
## Below I am defining my data matrix X and response vector y
## I am also randomly dividing my data into a training set of 70% and test set of 30%
X, y = Hyper.iloc[:,:-1], Hyper.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)


# In[6]:


## Here I will scale my data to a 0 to 1 scale. This is important for ANN and SVM. Less so for other algorithms. Note here we are
##scaling the features not the labels!
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_s = scaler.fit(X_train).transform(X_train)
X_test_s = scaler.fit(X_test).transform(X_test)


# In[7]:


##Here I am importing the necessary libraries to plot the lambda lambda coreelation for inspection of band autocorrelation
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')
import seaborn as sns
corr = X.corr()


# In[8]:


##I am specifying the parameters for the Lambda/Lambda plot.
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
sns.set(style="white")
f, ax = plt.subplots(figsize=(15, 15))
cmap = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1, center=0,
            square=True, linewidths=.7, cbar_kws={"shrink": 0.7})


# In[9]:


##Now we can start comparing the different ML classification algorithms. First I call the algorithm, then I run it on the scaled test data
## I then assess its perfomance using the accuracy measurement (both on test and training data), and the confusuin matrix.
##I will start with the most simple straight forward algorithm that is the decision tree.
from sklearn.tree import DecisionTreeClassifier
dtree_model = DecisionTreeClassifier(max_depth = 10).fit(X_train_s, y_train)
dtree_predictions = dtree_model.predict(X_test_s)
cm_DT = confusion_matrix(y_test, dtree_predictions)
print cm_DT


# In[10]:


##Here I look at the overall classification accurcay; both on the training and test sets
print('Accuracy of DT10, on the training set: {:.3f}'.format(dtree_model.score(X_train_s, y_train)))
print('Accuracy of DT10, on the test set: {:.3f}'.format(dtree_model.score(X_test_s, y_test)))


# In[11]:


##Let's allow a deeper tree to devlop and compare the outcome. Instead of pruning at 10 nodes, I will do 15 nodes.
from sklearn.tree import DecisionTreeClassifier
dtree_model15 = DecisionTreeClassifier(max_depth = 15).fit(X_train_s, y_train)
dtree_predictions15 = dtree_model15.predict(X_test_s)
cm_DT15 = confusion_matrix(y_test, dtree_predictions15)
print cm_DT15
print('Accuracy of DT15, on the training set: {:.3f}'.format(dtree_model15.score(X_train_s, y_train)))
print('Accuracy of DT15, on the test set: {:.3f}'.format(dtree_model15.score(X_test_s, y_test)))


# In[12]:


##allowing more nodes is showing an overfitting issue (above example) where the model does very well on the training set (~0.99%) and poorly on the test set (0.75%).
## We can conclude that we need smaller than 15 nodes. In other words, we need to prune the tree back to 10 or so nodes.
##We can also impose a threshold for a node split as a measure to avoid overfitting. Let's force a minimum of 100 sample per leaf.
dtree_leaf = DecisionTreeClassifier(min_samples_split=100).fit(X_train_s, y_train)
dtree_pred_leaf = dtree_leaf.predict(X_test_s)
cm_DT_leaf = confusion_matrix(y_test, dtree_pred_leaf)
print cm_DT_leaf
print('Accuracy of DT_leaf, on the training set: {:.3f}'.format(dtree_leaf.score(X_train_s, y_train)))
print('Accuracy of DT_leaf, on the test set: {:.3f}'.format(dtree_leaf.score(X_test_s, y_test)))


# In[13]:


##Now let's allow the decision to develop until pure leaf stage (no pruning and no imposed samples per pure leaf)
dtree_model_def = DecisionTreeClassifier().fit(X_train_s, y_train)
dtree_pred_def = dtree_model_def.predict(X_test_s)
cm_DT_def = confusion_matrix(y_test, dtree_pred_def)
print cm_DT_def
print('Accuracy of the default DT, on the training set: {:.3f}'.format(dtree_model_def.score(X_train_s, y_train)))
print('Accuracy of DT15, on the test set: {:.3f}'.format(dtree_model_def.score(X_test_s, y_test)))


# In[14]:


##The above example is the pefect case of overfitting. The tree labels the training data perfectly but does poorly on the test set.
## Now we will try Random Forest algorithm that average the prediction of multiple trees. We will impose 100 trees with 15 nodes.
from sklearn.ensemble import RandomForestClassifier
forest20 = RandomForestClassifier(n_estimators=100, max_depth = 20, random_state=0)
forest20.fit(X_train_s, y_train)
print('Accuracy of the Random Forest20, on the training set: {:.3f}'.format(forest20.score(X_train_s, y_train)))
print('Accuracy of the Random Forest20, on the test set: {:.3f}'.format(forest20.score(X_test_s, y_test)))


# In[16]:


forest20_predictions= forest20.predict(X_test_s)
cm_RF20 = confusion_matrix(y_test, forest20_predictions)
print cm_RF20


# In[17]:


##The above is showing an overfitting issue. let's reduce prune the trees to 15 nodes!
forest15 = RandomForestClassifier(n_estimators=100, max_depth = 15, random_state=0)
forest15.fit(X_train_s, y_train)
print('Accuracy of the Random Forest15, on the training set: {:.3f}'.format(forest15.score(X_train_s, y_train)))
print('Accuracy of the Random Forest15, on the test set: {:.3f}'.format(forest15.score(X_test_s, y_test)))
forest15_predictions= forest15.predict(X_test_s)
cm_RF15 = confusion_matrix(y_test, forest15_predictions)
print cm_RF15


# In[18]:


##The above is showing an improvement however we still have an overfitting issue (significant difference between the training and test sets performance)
##Let's prune to 10 nodes.
forest10 = RandomForestClassifier(n_estimators=100, max_depth = 10, random_state=0)
forest10.fit(X_train_s, y_train)
print('Accuracy of the Random Forest10, on the training set: {:.3f}'.format(forest10.score(X_train_s, y_train)))
print('Accuracy of the Random Forest10, on the test set: {:.3f}'.format(forest10.score(X_test_s, y_test)))
forest10_predictions= forest10.predict(X_test_s)
cm_RF10 = confusion_matrix(y_test, forest10_predictions)
print cm_RF10


# In[19]:


##Now let's use the raw data (unscaled) and see if we get any improvement.
forest_raw = RandomForestClassifier(n_estimators=100, max_depth = 10, random_state=0)
forest_raw.fit(X_train, y_train)
print('Accuracy of the Random Forest, on the training set: {:.3f}'.format(forest_raw.score(X_train, y_train)))
print('Accuracy of the Random Forest, on the test set: {:.3f}'.format(forest_raw.score(X_test, y_test)))
forest_raw_predictions= forest_raw.predict(X_test)
cm_RF_raw = confusion_matrix(y_test, forest_raw_predictions)
print cm_RF_raw


# In[20]:


##Here we show that scaling does not impact the performance of trees! verall, it seems we are getting about 80% accuracy
##using the RF algorithm. Now lets move on to K nearest classifier. 
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=20, n_jobs=-1)
knn.fit(X_train, y_train)
print('Accuracy of KNN n-5, on the training set: {:.3f}'.format(knn.score(X_train, y_train)))
print('Accuracy of KNN n-5, on the test set: {:.3f}'.format(knn.score(X_test, y_test)))
KNN_predictions= knn.predict(X_test)
cm_knn= confusion_matrix(y_test, KNN_predictions)
print cm_knn


# In[21]:


##The accuracy is somewhat poor. Let's try 30 neighbors instead of 20. Jobs=-1 allow the algorithm to use my computer processor multicores.
knn30 = KNeighborsClassifier(n_neighbors=30, n_jobs=-1)
knn30.fit(X_train, y_train)
print('Accuracy of KNN n-30, on the training set: {:.3f}'.format(knn30.score(X_train, y_train)))
print('Accuracy of KNN n-30, on the test set: {:.3f}'.format(knn30.score(X_test, y_test)))
KNN30_predictions= knn30.predict(X_test)
cm_knn30= confusion_matrix(y_test, KNN30_predictions)
print cm_knn30


# In[22]:


## We can conclude an overall accuracy of 78%. The next classifier we are trying is the NAive Bayes.
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB().fit(X_train, y_train)
gnb_predictions = gnb.predict(X_test)
print('Accuracy of NB on the training set: {:.3f}'.format(gnb.score(X_train, y_train)))
print('Accuracy of NB on the test set: {:.3f}'.format(gnb.score(X_test, y_test)))
cm_gnb = confusion_matrix(y_test, gnb_predictions)
print cm_gnb


# In[23]:


##The Naive Bayes classifyer seems to do very poorly on both the trianing and test datasets. Now we will try
##Logistic regression algorithm.
from sklearn import linear_model
LR=linear_model.LogisticRegression().fit(X_train, y_train)
LR_predictions = LR.predict(X_test)
print('Accuracy of LR on the training set: {:.3f}'.format(LR.score(X_train, y_train)))
print('Accuracy of LR on the test set: {:.3f}'.format(LR.score(X_test, y_test)))
cm_LR = confusion_matrix(y_test, LR_predictions)
print cm_LR


# In[24]:


##The logistic regression model seems to perform well both on the training and test sets. We can adjust the C parameter (default=1)
## and choose a smaller C value
LR2=linear_model.LogisticRegression(C=0.2).fit(X_train, y_train)
LR2_predictions = LR2.predict(X_test)
print('Accuracy of LR (C=0.2) on the training set: {:.3f}'.format(LR2.score(X_train, y_train)))
print('Accuracy of LR (C=0.2) on the test set: {:.3f}'.format(LR2.score(X_test, y_test)))
cm_LR2 = confusion_matrix(y_test, LR2_predictions)
print cm_LR2


# In[25]:


##We can use a C =50 to give more weight to the correctly classfied samples and see the the impact on the performance!
LR50=linear_model.LogisticRegression(C=50).fit(X_train, y_train)
LR50_predictions = LR50.predict(X_test)
print('Accuracy of LR (C=50) on the training set: {:.3f}'.format(LR50.score(X_train, y_train)))
print('Accuracy of LR (C=50) on the test set: {:.3f}'.format(LR50.score(X_test, y_test)))
cm_LR50 = confusion_matrix(y_test, LR50_predictions)
print cm_LR50


# In[26]:


##Overall, the LR performed better than the prvious algorithms (stable with regard to C)! 
##Now let's try Support Vector Machine or SVM. We will use the scaled data. 
from sklearn.svm import SVC
svm_linear = SVC(kernel = 'linear', C = 1).fit(X_train_s, y_train)
svm_predictions = svm_linear.predict(X_test_s)
print('Accuracy of SVM on the training set: {:.3f}'.format(svm_linear.score(X_train_s, y_train)))
print('Accuracy of SVM on the test set: {:.3f}'.format(svm_linear.score(X_test_s, y_test)))
cm_svm = confusion_matrix(y_test, svm_predictions)
print cm_svm


# In[27]:


##The above SVM model outperformed the previous models but similar to the LR. Next we will try the ANN. The Multi Layer Perceptron 
### or MLP is an example of Neural Nets with forword propagation.
from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(random_state=42)
mlp.fit(X_train_s, y_train)
print('Accuracy of MLP on the training subset: {:.3f}'.format(mlp.score(X_train_s, y_train)))
print('Accuracy of MLP on the test subset: {:.3f}'.format(mlp.score(X_test_s, y_test)))
ANN_predictions = mlp.predict(X_test_s)
cm_mlp = confusion_matrix(y_test, ANN_predictions)
print cm_mlp


# In[28]:


##MLP shows good accuracy but potential overfitting. Let's adjust the learning rate and observe the difference.
mlp01 = MLPClassifier(learning_rate_init=0.001, random_state=42)
mlp01.fit(X_train_s, y_train)
print('Accuracy of MLP (learning rate = 0.1) on the training subset: {:.3f}'.format(mlp01.score(X_train_s, y_train)))
print('Accuracy of MLP (learning rate = 0.1) on the test subset: {:.3f}'.format(mlp01.score(X_test_s, y_test)))
ANN01_predictions = mlp01.predict(X_test_s)
cm_mlp01 = confusion_matrix(y_test, ANN01_predictions)
print cm_mlp01


# In[29]:


mlp1000 = MLPClassifier(learning_rate_init=0.001, max_iter=1000, random_state=42)
mlp1000.fit(X_train_s, y_train)
print('Accuracy of MLP (1000 iterations) on the training subset: {:.3f}'.format(mlp1000.score(X_train_s, y_train)))
print('Accuracy of MLP (1000 iterations) on the test subset: {:.3f}'.format(mlp1000.score(X_test_s, y_test)))
ANN1000_predictions = mlp1000.predict(X_test_s)
cm_mlp1000 = confusion_matrix(y_test, ANN1000_predictions)
print cm_mlp1000



# In[39]:


##Increasing the number of iterations for the MLP does not impact the performance. Overall, MLP shows promising results. 
##The next algorithm is Gradient Boosting trees.
from sklearn.ensemble import GradientBoostingClassifier
Boost = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, min_samples_split=1000, max_depth=10, random_state=42, max_features=16).fit(X_train, y_train)
print('Accuracy of Gradient Boosting on the training subset: {:.3f}'.format(Boost.score(X_train, y_train)))
print('Accuracy of Gradient Boosting on the test subset: {:.3f}'.format(Boost.score(X_test, y_test)))
Boost_predictions = Boost.predict(X_test)
cm_Boost = confusion_matrix(y_test, Boost_predictions)
print cm_Boost


# In[44]:


## Now I will attempt to reduce the dimesnion of the data by ranking the bands based on their importance based on Random Forest.
##This method is more powerful than relying on univariate statistcis such as percentile or variance (Mueller and Guido 2016).
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')
from sklearn.ensemble import ExtraTreesClassifier


# In[71]:


clf = ExtraTreesClassifier()
clf = clf.fit(X, y)
importances = clf.feature_importances_
std = np.std([tree.feature_importances_ for tree in forest.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]
print("Feature ranking:")
for f in range(X.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))
    
plt.figure()
plt.title("Feature importances")
plt.bar(range(X.shape[1]), importances[indices],
       color="r", yerr=std[indices], align="center")
plt.xticks(range(X.shape[1]), indices)
plt.xlim([-1, X.shape[1]])
plt.show()



# In[72]:


from sklearn.feature_selection import SelectFromModel
model = SelectFromModel(clf, 0.010, prefit=True)
X_new = model.transform(X)
X_new.shape


# In[93]:


clf_red = ExtraTreesClassifier()
clf_red = clf_red.fit(X_new, y)
importances = clf_red.feature_importances_
std = np.std([tree.feature_importances_ for tree in forest.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]
bandnames_select=[X.columns[i] for i in indices]
print bandnames_select
print("Feature ranking:")
for f in range(X_new.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))
    
plt.figure()
plt.title("Feature importances")
plt.bar(range(X_new.shape[1]), importances[indices],
       color="r", yerr=std[indices], align="center")
plt.xticks(range(X_new.shape[1]), indices)
plt.xlim([-1, X_new.shape[1]])
plt.show()


# In[75]:


## here are the top important 15 bands retained; #8 = b234 (see all bands ranking above for band IdsB1 to B240), 0 is band B1; 9 is band183; 2 is band180; 4 is B179; 
## and 14 is band144; 7 is B141; 10 is B123; 3 is B209; 5 is B97; 1 is B107; 11 is B197; 12 is B212; 6 is B205 and 13 is B19.
##we notice the importnace and dominace of the NIR bands. B1 is 395nm and B240 is 900nm (increments of 2.1)
## Now we will rerun the predictions with this smaller set of features. First we scale the new data (X_new). We will focus on
##RF, Boosting, SVM, MLP and LR based on the previous comparison using all data.

X_new_train, X_new_test, y_train, y_test = train_test_split(X_new, y, test_size=0.3, random_state=1)
scaler = StandardScaler()
X_new_train_s = scaler.fit(X_new_train).transform(X_new_train)
X_new_test_s = scaler.fit(X_new_test).transform(X_new_test)


# In[77]:


##Now let's rerun the DT with the reduce dimension data (15 bands only)
dtree_modN = DecisionTreeClassifier(max_depth = 10).fit(X_new_train_s, y_train)
dtree_predN = dtree_modN.predict(X_new_test_s)
cm_DTN = confusion_matrix(y_test, dtree_predN)
print cm_DTN
print('Accuracy of DT X new, on the training set: {:.3f}'.format(dtree_modN.score(X_new_train_s, y_train)))
print('Accuracy of DT X new, on the test set: {:.3f}'.format(dtree_modN.score(X_new_test_s, y_test)))


# In[78]:


##We do the same with RF
forest_red = RandomForestClassifier(n_estimators=100, max_depth = 10, random_state=0)
forest_red.fit(X_new_train_s, y_train)
print('Accuracy of the Random Forest with 15 bands, on the training set: {:.3f}'.format(forest_red.score(X_new_train_s, y_train)))
print('Accuracy of the Random Forest with 15 bands, on the test set: {:.3f}'.format(forest_red.score(X_new_test_s, y_test)))
forest_red_predictions= forest_red.predict(X_new_test_s)
cm_RF_red = confusion_matrix(y_test, forest_red_predictions)
print cm_RF_red


# In[79]:


## Now we will run the LR
LR_red=linear_model.LogisticRegression(C=50).fit(X_new_train, y_train)
LR_red_predictions = LR_red.predict(X_new_test)
print('Accuracy of LR with 15 bands (C=50) on the training set: {:.3f}'.format(LR_red.score(X_new_train, y_train)))
print('Accuracy of LR with 15 bands (C=50) on the test set: {:.3f}'.format(LR_red.score(X_new_test, y_test)))
cm_LR_red = confusion_matrix(y_test, LR_red_predictions)
print cm_LR_red


# In[81]:


##We recall here that the accuracy on the hidden (test set) data was 90% when we run LR on the entire data.
##Now we will run the MLP
mlp_red = MLPClassifier(learning_rate_init=0.001, max_iter=1000, random_state=42)
mlp_red.fit(X_new_train_s, y_train)
print('Accuracy of MLP with 15 bands (1000 iterations) on the training subset: {:.3f}'.format(mlp_red.score(X_new_train_s, y_train)))
print('Accuracy of MLP with 15 bands (1000 iterations) on the test subset: {:.3f}'.format(mlp_red.score(X_new_test_s, y_test)))
MLP_red_predictions = mlp_red.predict(X_new_test_s)
cm_mlp_red = confusion_matrix(y_test, MLP_red_predictions)
print cm_mlp_red


# In[82]:


## The MLP seems to be stable with a moderate accuracy.
##Now we run the SVM
svm_red = SVC(kernel = 'linear', C = 1).fit(X_new_train_s, y_train)
svm_red_predictions = svm_red.predict(X_new_test_s)
print('Accuracy of SVM  with 15 bands on the training set: {:.3f}'.format(svm_red.score(X_new_train_s, y_train)))
print('Accuracy of SVM with 15 bands on the test set: {:.3f}'.format(svm_red.score(X_new_test_s, y_test)))
cm_svm_red = confusion_matrix(y_test, svm_red_predictions)
print cm_svm_red


# In[87]:


X.shape


# In[83]:


##The SVM is slightly outperformed by the MLP (Neural nets)
##The last algorithm we will run with this reduced data is the Gradient Boosting (GB)
GB = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, min_samples_split=200, max_depth=10, random_state=42).fit(X_new_train, y_train)
print('Accuracy of Gradient Boosting with 15 bands on the training subset: {:.3f}'.format(GB.score(X_new_train, y_train)))
print('Accuracy of Gradient Boosting with 15 bands on the test subset: {:.3f}'.format(GB.score(X_new_test, y_test)))
GB_predictions = GB.predict(X_new_test)
cm_GB = confusion_matrix(y_test, GB_predictions)
print cm_GB


# In[91]:


## Overall, the SVM, MLP, and LR shows the best accuracies (less prone to overfitting)
## Now we will try a different method for data reduction using the K best features (based on variance) then we will compare
##the best algorithms again.
import sklearn.feature_selection
selectkb=sklearn.feature_selection.SelectKBest(k=15)
select_features= selectkb.fit(X_train, y_train)
indices_selected=select_features.get_support(indices=True)
colnames_select=[X.columns[i] for i in indices_selected]
X_train_selected=X_train[colnames_select]
X_test_selected=X_test[colnames_select]
print colnames_select



# In[95]:


##15 best indices are dominated by red bands and only one band in NIR.
from sklearn.feature_selection import chi2
selectkb=sklearn.feature_selection.SelectKBest(chi2, k=15)
select_features= selectkb.fit(X_train, y_train)
indices_selected=select_features.get_support(indices=True)
colnames_select=[X.columns[i] for i in indices_selected]
X_train_selected=X_train[colnames_select]
X_test_selected=X_test[colnames_select]
print colnames_select


# In[97]:


##sklearn.feature_selection.mutual_info_classif(X, y, discrete_features=’auto’, n_neighbors=50, copy=True, random_state=None)
from sklearn.feature_selection import mutual_info_classif


# In[105]:


##This is a non parametric method that looks at the degree of dependency between each attribue (band) and the target (N streess).
##values are between 0 and 1 with zero means total independece and 1 means strong dependency. Very interesting is the strong 
##relationship between Violet light (B397) and N stress that I did not expect. Other interesting bands are green (570),the red around 700nm and the NIR

sklearn.feature_selection.mutual_info_classif(X, y, n_neighbors=10, copy=True, random_state=None)


# In[106]:


##Now we will explore how the PCA can be used to reduce the dimesnion of the data. We wil then apply the different algorithms
## to the PCA transformed data (PCA components). First we will run the SVM
from sklearn.decomposition import PCA
pca = PCA(n_components=6) #I am chosing 6 to start with
pca.fit(X_train) ##we conduct the PCA on the training set only to avoid overfitting
X_t_train = pca.transform(X_train)
X_t_test = pca.transform(X_test)
svm_pca = SVC(kernel = 'linear', C = 1).fit(X_t_train, y_train)
svm_pca_predictions = svm_pca.predict(X_t_test)
print('Accuracy of SVM  with 6 PCA on the training set: {:.3f}'.format(svm_pca.score(X_t_train, y_train)))
print('Accuracy of SVM with 6 PCA on the test set: {:.3f}'.format(svm_pca.score(X_t_test, y_test)))
cm_svm_pca6 = confusion_matrix(y_test, svm_pca_predictions)
print cm_svm_pca6



# In[107]:


##There is no evidence of overfitting but the performance is poor.
##Let's now try with 10pcas instead of 6
pca10 = PCA(n_components=10) #I am chosing 10 to start with
pca10.fit(X_train) ##we conduct the PCA on the training set only to avoid overfitting
X_t_train10 = pca10.transform(X_train)
X_t_test10 = pca10.transform(X_test)
svm_pca10 = SVC(kernel = 'linear', C = 1).fit(X_t_train10, y_train)
svm_pca10_predictions = svm_pca10.predict(X_t_test10)
print('Accuracy of SVM  with 10 PCA on the training set: {:.3f}'.format(svm_pca10.score(X_t_train10, y_train)))
print('Accuracy of SVM with 10 PCA on the test set: {:.3f}'.format(svm_pca10.score(X_t_test10, y_test)))
cm_svm_pca10 = confusion_matrix(y_test, svm_pca10_predictions)
print cm_svm_pca10


# In[111]:


##There is almost no improvement so we will move to the MLP
pca3 = PCA(n_components=3) #I am chosing 3 pcs to start with
pca3.fit(X_train) ##we conduct the PCA on the training set only to avoid overfitting
X_t_train3 = pca3.transform(X_train)
X_t_test3 = pca3.transform(X_test)
mlp_pca3 = MLPClassifier(learning_rate_init=0.001, max_iter=1000)
mlp_pca3.fit(X_t_train3, y_train)
print('Accuracy of MLP with 3 pcs (1000 iterations) on the training subset: {:.3f}'.format(mlp_pca3.score(X_t_train3, y_train)))
print('Accuracy of MLP with 3 pcs (1000 iterations) on the test subset: {:.3f}'.format(mlp_pca3.score(X_t_test3, y_test)))
MLP_pca3_predictions = mlp_pca3.predict(X_t_test3)
cm_pca3 = confusion_matrix(y_test, MLP_pca3_predictions)
print cm_pca3


# In[112]:


## now we will retry with 6pcs.
pca6 = PCA(n_components=6) #I am chosing 6 pcs to start with
pca6.fit(X_train) ##we conduct the PCA on the training set only to avoid overfitting
X_t_train6 = pca6.transform(X_train)
X_t_test6 = pca6.transform(X_test)
mlp_pca6 = MLPClassifier(learning_rate_init=0.001, max_iter=1000)
mlp_pca6.fit(X_t_train6, y_train)
print('Accuracy of MLP with 6 pcs (1000 iterations) on the training subset: {:.3f}'.format(mlp_pca6.score(X_t_train6, y_train)))
print('Accuracy of MLP with 6 pcs (1000 iterations) on the test subset: {:.3f}'.format(mlp_pca6.score(X_t_test6, y_test)))
MLP_pca6_predictions = mlp_pca6.predict(X_t_test6)
cm_pca6 = confusion_matrix(y_test, MLP_pca6_predictions)
print cm_pca6


# In[113]:


## Now we will read a short version of the data that consists of 8 bands that I selected based on the hyperspectral
##spectral signature. One violet, one green, one red, and 3 NIR including one in the water absorbance window.
Aicam=pd.read_csv("C:\Users\garag\Google Drive\Hyperspectral\Auto_bandAicam.csv")


# In[114]:


##Let's look at its size
Aicam.shape


# In[119]:


## let's split the data into features and target or label. We will also split it into 30% test set and 70% training set
X, y = Aicam.iloc[:,:-1], Aicam.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)


# In[122]:


##Now we need to scale it so it can be used in MLP and SVM that are sensitive to unscaled data.
scaler = StandardScaler()
X_train_sc = scaler.fit(X_train).transform(X_train)
X_test_sc = scaler.fit(X_test).transform(X_test)


# In[123]:


##Now we can examine the correlation plot. 
corr1 = X.corr()
mask = np.zeros_like(corr1, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
sns.set(style="white")
f, ax = plt.subplots(figsize=(15, 15))
cmap = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(corr1, mask=mask, cmap=cmap, vmax=1, center=0,
            square=True, linewidths=.7, cbar_kws={"shrink": 0.7})


# In[124]:


## Decision tree
dt_model = DecisionTreeClassifier().fit(X_train_sc, y_train)
dt_predictions = dt_model.predict(X_test_sc)
cm_dt = confusion_matrix(y_test, dt_predictions)
print('Accuracy of DT, on the training set: {:.3f}'.format(dt_model.score(X_train_sc, y_train)))
print('Accuracy of DT15, on the test set: {:.3f}'.format(dt_model.score(X_test_sc, y_test)))
print cm_dt


# In[125]:


##The above example is a typical case of overfitting... We will impose restrictions on the DT to avoid this issue.
dt5_model = DecisionTreeClassifier(max_depth = 5).fit(X_train_sc, y_train)
dt5_predictions = dt5_model.predict(X_test_sc)
cm_dt5 = confusion_matrix(y_test, dt5_predictions)
print('Accuracy of DT 5, on the training set: {:.3f}'.format(dt5_model.score(X_train_sc, y_train)))
print('Accuracy of DT 5, on the test set: {:.3f}'.format(dt5_model.score(X_test_sc, y_test)))
print cm_dt5


# In[126]:


##There is no overfitting but poor performance. Let's consider Ensemble methods (Random Forest and Gradient Boosting)
##We run Random Forest first with 100 iterations and 5 nodes max.
forest1 = RandomForestClassifier(n_estimators=100, max_depth = 5, random_state=0)
forest1.fit(X_train_sc, y_train)
print('Accuracy of the Random Forest, on the training set: {:.3f}'.format(forest1.score(X_train_sc, y_train)))
print('Accuracy of the Random Forest, on the test set: {:.3f}'.format(forest1.score(X_test_sc, y_test)))
forest1_predictions= forest1.predict(X_test_sc)
cm_RF1 = confusion_matrix(y_test, forest1_predictions)
print cm_RF1


# In[127]:


##Here we see a slightly better performace than one single tree but still poor overall performance! If we dont prune the trees
##it will result in overfitting as below
forest2 = RandomForestClassifier(n_estimators=100, random_state=0)
forest2.fit(X_train_sc, y_train)
print('Accuracy of the Random Forest, on the training set: {:.3f}'.format(forest2.score(X_train_sc, y_train)))
print('Accuracy of the Random Forest, on the test set: {:.3f}'.format(forest2.score(X_test_sc, y_test)))
forest2_predictions= forest2.predict(X_test_sc)
cm_RF2 = confusion_matrix(y_test, forest2_predictions)
print cm_RF2


# In[128]:


## Now let's compare the importance of the different bands using ensemble methods.
clf1 = ExtraTreesClassifier()
clf1 = clf1.fit(X, y)
importances = clf1.feature_importances_
std = np.std([tree.feature_importances_ for tree in forest.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]
print("Feature ranking:")
for f in range(X.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))
    
plt.figure()
plt.title("Feature importances")
plt.bar(range(X.shape[1]), importances[indices],
       color="r", yerr=std[indices], align="center")
plt.xticks(range(X.shape[1]), indices)
plt.xlim([-1, X.shape[1]])
plt.show()


# In[129]:


#It seems that Band397 is the most significant followed by the red and the green then the NIR3.
##Let's run the GB classification.
GB = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, min_samples_split=1000, random_state=42).fit(X_train_sc, y_train)
print('Accuracy of Gradient Boosting on the training subset: {:.3f}'.format(GB.score(X_train_sc, y_train)))
print('Accuracy of Gradient Boosting on the test subset: {:.3f}'.format(GB.score(X_test_sc, y_test)))
Boost_predictions = GB.predict(X_test_sc)
cm_Boost = confusion_matrix(y_test, Boost_predictions)
print cm_Boost


# In[130]:


#GB shows a slight improvement over RF taking advantage of the indepedent trees.
##Now will run the LR
Logit=linear_model.LogisticRegression(C=50).fit(X_train_sc, y_train)
Logit_predictions = Logit.predict(X_test_sc)
print('Accuracy of LR (C=50) on the training set: {:.3f}'.format(Logit.score(X_train_sc, y_train)))
print('Accuracy of LR (C=50) on the test set: {:.3f}'.format(Logit.score(X_test_sc, y_test)))
cm_Logit = confusion_matrix(y_test, Logit_predictions)
print cm_Logit


# In[131]:


## and the MLP
mlp1 = MLPClassifier(learning_rate_init=0.001, max_iter=1000, random_state=42)
mlp1.fit(X_train_sc, y_train)
print('Accuracy of MLP with 8 bands (1000 iterations) on the training subset: {:.3f}'.format(mlp1.score(X_train_sc, y_train)))
print('Accuracy of MLP with 8 bands (1000 iterations) on the test subset: {:.3f}'.format(mlp1.score(X_test_sc, y_test)))
mlp_predictions = mlp1.predict(X_test_sc)
cm_mlp1 = confusion_matrix(y_test, mlp_predictions)
print cm_mlp1


# In[133]:


# and the SVM
svm8 = SVC(kernel = 'linear', C = 1).fit(X_train_sc, y_train)
svm8_predictions = svm8.predict(X_test_sc)
print('Accuracy of SVM  with 8 bands on the training set: {:.3f}'.format(svm8.score(X_train_sc, y_train)))
print('Accuracy of SVM with 8 bands on the test set: {:.3f}'.format(svm8.score(X_test_sc, y_test)))
cm_svm8 = confusion_matrix(y_test, svm8_predictions)
print cm_svm8


# In[134]:


##Let's try KNN
knn = KNeighborsClassifier(n_neighbors=30)
knn.fit(X_train_sc, y_train)
print('Accuracy of KNN n-30, on the training set: {:.3f}'.format(knn.score(X_train_sc, y_train)))
print('Accuracy of KNN n-30, on the test set: {:.3f}'.format(knn.score(X_test_sc, y_test)))
KNN_predictions= knn.predict(X_test_sc)
cm_knn= confusion_matrix(y_test, KNN_predictions)
print cm_knn


# In[ ]:


##Based on the result above, Gradient Boosting and Multi-Layer perceptron outperformed all other algorithms with
## 76% amd 75% accuracies respectively.

