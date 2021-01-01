#!/usr/bin/env python
# coding: utf-8

# In[8]:


import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy import stats
import numpy as np
from statsmodels.graphics.gofplots import qqplot


# In[9]:


heart_disease = pd.read_csv('heart_statlog_cleveland_hungary_final.csv')


# In[10]:


heart_disease.shape


# In[11]:


heart_disease.dtypes


# In[12]:


heart_disease.head()


# In[13]:


heart_disease['target'].value_counts()
#1 : heart disease, 0: normal


# In[14]:


heart_disease.describe()


# In[15]:


heart_disease.groupby('target').mean()


# In[16]:


#Correlation matrix
heart_disease.corr()


# In[17]:


def describe_cont_feature(feature):
    print('\n*** Results for {} ***'.format(feature))
    print(heart_disease.groupby('target')[feature].describe())
    print(ttest(feature))
    
def ttest(feature):
    disease_positive = heart_disease[heart_disease['target']==1][feature]
    disease_negative = heart_disease[heart_disease['target']==0][feature]
    tstat, pval = stats.ttest_ind(disease_positive, disease_negative, equal_var=False)
    print('t-statistic: {:.1f}, p-value: {:.3}'.format(tstat, pval))


# In[18]:


features = list(heart_disease.columns)
features.remove('target')
features


# In[ ]:





# In[19]:


# Look at the distribution of each feature at each level of the target variable
for feature in features:
    describe_cont_feature(feature)


# ## Plot continous features

# In[20]:


# Plot overlaid histograms for continuous features
non_continous_features = ['sex', 'chest pain type','fasting blood sugar','resting ecg','exercise angina','ST slope']
continous_features = features
for feature in non_continous_features:
    continous_features.remove(feature)
print(continous_features)


# In[21]:


import warnings
warnings.filterwarnings("ignore")
for i in continous_features:
    normal = list(heart_disease[heart_disease['target'] == 0][i].dropna())
    disease = list(heart_disease[heart_disease['target'] == 1][i].dropna())
    xmin = min(min(normal), min(disease))
    xmax = max(max(normal), max(disease))
    width = (xmax - xmin) / 40
    sns.distplot(normal, color='g', kde=False, bins=np.arange(xmin, xmax, width))
    sns.distplot(disease, color='r', kde=False, bins=np.arange(xmin, xmax, width))
    plt.legend(['normal', 'heart disease'])
    plt.title('Overlaid histogram for {}'.format(i))
    plt.show()


# ## Plot categorical features 

# In[22]:


# Generate categorical plots for ordinal features
for col in non_continous_features:
    sns.catplot(x=col, y='target', data=heart_disease, kind='point', aspect=1, )
    plt.ylim(0, 1)


# In[23]:


for col in non_continous_features:
    heart_disease_copy = heart_disease.filter(['target',col], axis=1)
    #heart_disease_copy = heart_disease['target', col]
    
    print(heart_disease_copy.groupby(col).mean())


# ## Detect and clean outliers 

# In[24]:


def detect_outlier(feature):
    outliers = []
    data = heart_disease[feature]
    mean = np.mean(data)
    std =np.std(data)
    
    
    for y in data:
        z_score= (y - mean)/std 
        if np.abs(z_score) > 3:
            outliers.append(y)
    print('\nOutlier caps for {}:'.format(feature))
    print('  --95p: {:.1f} / {} values exceed that'.format(data.quantile(.95),
                                                             len([i for i in data
                                                                  if i > data.quantile(.95)])))
    print('  --3sd: {:.1f} / {} values exceed that'.format(mean + 3*(std), len(outliers)))
    print('  --99p: {:.1f} / {} values exceed that'.format(data.quantile(.99),
                                                           len([i for i in data
                                                                if i > data.quantile(.99)])))


# In[25]:


for feature in continous_features:
    detect_outlier(feature)


# In[26]:


heart_disease.describe()


# In[27]:


# Cap features
heart_disease['age'].clip(upper=heart_disease['age'].quantile(.99), inplace=True)
heart_disease['resting bp s'].clip(upper=heart_disease['resting bp s'].quantile(.99), inplace=True)
heart_disease['cholesterol'].clip(upper=heart_disease['cholesterol'].quantile(.99), inplace=True)
heart_disease['max heart rate'].clip(upper=heart_disease['max heart rate'].quantile(.99), inplace=True)
heart_disease['oldpeak'].clip(upper=heart_disease['oldpeak'].quantile(.99), inplace=True)


# In[28]:


heart_disease.describe()


# In[ ]:





# ## Transform Skewed Features

# In[29]:


for feature in continous_features:
    sns.distplot(heart_disease[feature], kde=False)
    plt.title('Histogram for {}'.format(feature))
    plt.show()


# In[30]:


# Generate QQ plots
for i in [0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
    data_t = heart_disease['resting bp s']**(1/i)
    qqplot(data_t, line='s')
    plt.title("Transformation: 1/{}".format(str(i)))

#1/2


# In[31]:


# Generate QQ plots
for i in [0.25, 0.5, 0.75, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
    data_t = heart_disease['cholesterol']**(1/i)
    qqplot(data_t, line='s')
    plt.title("Transformation: 1/{}".format(str(i)))

#1/0.75


# In[32]:


# Generate QQ plots
for i in [0.25, 0.5, 0.75, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
    data_t = heart_disease['oldpeak']**(1/i)
    qqplot(data_t, line='s')
    plt.title("Transformation: 1/{}".format(str(i)))

#1/1


# In[33]:


# Create the new transformed feature
heart_disease['resting bp s'] = heart_disease['resting bp s'].apply(lambda x: x**(1/2))
heart_disease['cholesterol'] = heart_disease['cholesterol'].apply(lambda x: x**(4/3))


# In[34]:


heart_disease.head()


# ## Sharang's Analysis

# In[35]:


import copy
heart_disease_truncated = copy.deepcopy(heart_disease)
del heart_disease_truncated['resting bp s']
del heart_disease_truncated['resting ecg']


# In[36]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle

X = heart_disease.iloc[:, :-1].values
y = heart_disease.iloc[:, -1].values

X_train_full, X_test_full, y_train_full, y_test_full = train_test_split(X,y,test_size = 0.4, random_state = 1)
X_val_full, X_test_full, y_val_full, y_test_full = train_test_split(X_test_full,y_test_full,test_size = 0.5, random_state = 1)


sc = StandardScaler()
X_train_full = sc.fit_transform(X_train_full)
X_test_full = sc.transform(X_test_full)
X_val_full = sc.transform(X_val_full)

with open(r"train_features.pickle", "wb") as output_file:
    pickle.dump(X_train_full, output_file)
with open(r"test_features.pickle", "wb") as output_file:
    pickle.dump(X_test_full, output_file)
with open(r"val_features.pickle", "wb") as output_file:
    pickle.dump(X_val_full, output_file)

with open(r"train_labels.pickle", "wb") as output_file:
    pickle.dump(y_train_full, output_file)
with open(r"test_labels.pickle", "wb") as output_file:
    pickle.dump(y_test_full, output_file)
with open(r"val_labels.pickle", "wb") as output_file:
    pickle.dump(y_val_full, output_file)


# In[37]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle

X = heart_disease_truncated.iloc[:, :-1].values
y = heart_disease_truncated.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.4, random_state = 1)
X_val, X_test, y_val, y_test = train_test_split(X_test,y_test,test_size = 0.5, random_state = 1)


sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
X_val = sc.transform(X_val)

with open(r"train_features_truncated.pickle", "wb") as output_file:
    pickle.dump(X_train, output_file)
with open(r"test_features_truncated.pickle", "wb") as output_file:
    pickle.dump(X_test, output_file)
with open(r"val_features_truncated.pickle", "wb") as output_file:
    pickle.dump(X_val, output_file)

with open(r"train_labels_truncated.pickle", "wb") as output_file:
    pickle.dump(y_train, output_file)
with open(r"test_labels_truncated.pickle", "wb") as output_file:
    pickle.dump(y_test, output_file)
with open(r"val_labels_truncated.pickle", "wb") as output_file:
    pickle.dump(y_val, output_file)


# In[74]:


#import pickle

#with open(r"train_features.pickle", "rb") as input_file:
#    X_train = pickle.load(input_file)


# In[39]:


import joblib
from sklearn.model_selection import GridSearchCV
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)


# ## Hyperparameter tuning

# In[40]:


def print_results(results):
    print('BEST PARAMS: {}\n'.format(results.best_params_))

    means = results.cv_results_['mean_test_score']
    stds = results.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, results.cv_results_['params']):
        print('{} (+/-{}) for {}'.format(round(mean, 3), round(std * 2, 3), params))


# ## Logistic Regression

# In[75]:


from sklearn.linear_model import LogisticRegression


# ## Full Feature Set

# In[70]:


lr = LogisticRegression()
parameters = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]
}

cv = GridSearchCV(lr, parameters, cv=5)
cv.fit(X_train_full, y_train_full)

print_results(cv)


# ## Truncated Feature Set

# In[71]:


lr = LogisticRegression()
parameters = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]
}

cv = GridSearchCV(lr, parameters, cv=5)
cv.fit(X_train, y_train)

print_results(cv)


# In[72]:


cv.best_estimator_


# In[73]:


joblib.dump(cv.best_estimator_, 'LR_model.pkl')


# ## Inference 

# Using Logistic Regression, the best accuracy is obtained using the truncated feature set, with hyperparamerer C = 0.1

# ## Support Vector Machine

# In[80]:


from sklearn.svm import SVC


# ## Full feature set

# In[81]:


svc = SVC()
parameters = {
    'kernel': ['linear', 'rbf'],
    'C': [0.1, 1, 10]
}

cv = GridSearchCV(svc, parameters, cv=5)
cv.fit(X_train_full, y_train_full)

print_results(cv)


# ## Truncated feature set

# In[82]:


svc = SVC()
parameters = {
    'kernel': ['linear', 'rbf'],
    'C': [0.1, 1, 10]
}

cv = GridSearchCV(svc, parameters, cv=5)
cv.fit(X_train, y_train)

print_results(cv)


# In[78]:


cv.best_estimator_


# In[79]:


joblib.dump(cv.best_estimator_, 'SVM_model.pkl')


# ## Inference 

# Using Support Vector Machine, the best accuracy is obtained using the truncated feature set, with hyperparamerer C = 1 and kernel = rbf

# ## Multi Layer Perceptron

# In[83]:


from sklearn.neural_network import MLPClassifier


# ## Full feature set

# In[84]:


mlp = MLPClassifier()
parameters = {
    'hidden_layer_sizes': [(10,), (50,), (100,)],
    'activation': ['relu', 'tanh', 'logistic'],
    'learning_rate': ['constant', 'invscaling', 'adaptive']
}

cv = GridSearchCV(mlp, parameters, cv=5)
cv.fit(X_train_full, y_train_full)

print_results(cv)


# ## Truncated feature set

# In[85]:


mlp = MLPClassifier()
parameters = {
    'hidden_layer_sizes': [(10,), (50,), (100,)],
    'activation': ['relu', 'tanh', 'logistic'],
    'learning_rate': ['constant', 'invscaling', 'adaptive']
}

cv = GridSearchCV(mlp, parameters, cv=5)
cv.fit(X_train, y_train)

print_results(cv)


# In[88]:


cv.best_estimator_


# In[89]:


joblib.dump(cv.best_estimator_, 'MLP_model.pkl')


# ## Inference 

# Using Multi Layer Perceptron, the best accuracy is obtained using the truncated feature set, with hyperparamerer 'activation' = 'tanh', 'hidden_layer_sizes' = (50,), 'learning_rate' = 'adaptive'
# 
# 

# ## Random Forest

# In[49]:


from sklearn.ensemble import RandomForestClassifier


# ## Full feature set

# In[50]:


rf = RandomForestClassifier()
parameters = {
    'n_estimators': [5, 50, 250],
    'max_depth': [2, 4, 8, 16, 32, None]
}

cv = GridSearchCV(rf, parameters, cv=5)
cv.fit(X_train_full, y_train_full)

print_results(cv)


# In[51]:


cv.best_estimator_


# ## Truncated feature set

# In[52]:


rf = RandomForestClassifier()
parameters = {
    'n_estimators': [5, 50, 250],
    'max_depth': [2, 4, 8, 16, 32, None]
}

cv = GridSearchCV(rf, parameters, cv=5)
cv.fit(X_train, y_train)

print_results(cv)


# In[54]:


cv.best_estimator_


# In[55]:


joblib.dump(cv.best_estimator_, 'RF_model.pkl')


# ## Inference 

# Using Random Forests, the best accuracy is obtained using the truncated dataset with 'max_depth' = 16, 'n_estimators' = 250

# ## Gradient boosting

# In[44]:


from sklearn.ensemble import GradientBoostingClassifier


# ## Full feature set

# In[45]:


gb = GradientBoostingClassifier()
parameters = {
    'n_estimators': [5, 50, 100, 150, 200, 250, 500],
    'max_depth': [1, 2, 3, 4, 5, 7, 9],
    'learning_rate': [0.01, 0.1, 1, 10, 100]
}

cv = GridSearchCV(gb, parameters, cv=5)
cv.fit(X_train_full, y_train_full)

print_results(cv)


# best accuracy : 0.877 (+/-0.044) for {'learning_rate': 0.01, 'max_depth': 7, 'n_estimators': 500}

# ## Truncated feature set

# In[46]:


gb = GradientBoostingClassifier()
parameters = {
    'n_estimators': [5, 50, 100, 150, 200, 250, 500],
    'max_depth': [1, 2, 3, 4, 5, 7, 9],
    'learning_rate': [0.01, 0.1, 1, 10, 100]
}

cv = GridSearchCV(gb, parameters, cv=5)
cv.fit(X_train, y_train)

print_results(cv)


# BEST PARAMS: {'learning_rate': 1, 'max_depth': 9, 'n_estimators': 50}

# In[47]:


cv.best_estimator_


# In[48]:


joblib.dump(cv.best_estimator_, 'GBC_model.pkl')


# ## Inference 

# Best accuracy obtained with full dataset with 'learning_rate': 0.01, 'max_depth': 7, 'n_estimators': 500 or truncated dataset with 'learning_rate': 1, 'max_depth': 9, 'n_estimators': 50

# ### Compare model results and final model selection

# In[57]:


from sklearn.metrics import accuracy_score, precision_score, recall_score
from time import time


# In[60]:


models = {}

for mdl in ['LR', 'SVM', 'MLP', 'RF', 'GBC']:
    models[mdl] = joblib.load('{}_model.pkl'.format(mdl))


# In[61]:


models


# ### Evaluate models on the validation set

# In[62]:


def evaluate_model(name, model, features, labels):
    start = time()
    pred = model.predict(features)
    end = time()
    accuracy = round(accuracy_score(labels, pred), 3)
    precision = round(precision_score(labels, pred), 3)
    recall = round(recall_score(labels, pred), 3)
    print('{} -- Accuracy: {} / Precision: {} / Recall: {} / Latency: {}ms'.format(name,
                                                                                   accuracy,
                                                                                   precision,
                                                                                   recall,
                                                                                   round((end - start)*1000, 1)))


# In[64]:


for name, mdl in models.items():
    evaluate_model(name, mdl, X_val, y_val)


# ### Evaluate best model on test set

# In[66]:


evaluate_model('Random Forest', models['RF'], X_test, y_test)


# In[67]:


evaluate_model('Gradient Boost', models['GBC'], X_test, y_test)


# In[69]:


evaluate_model('Support Vector Machine', models['SVM'], X_test, y_test)


# In[68]:


evaluate_model('Multi Level Perceptron', models['MLP'], X_test, y_test)


# ## Test on 80:20 split

# In[70]:


from sklearn.metrics import classification_report 
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

X = heart_disease_truncated.iloc[:, :-1].values
y = heart_disease_truncated.iloc[:, -1].values

x_train, x_test, y_train, y_test = train_test_split(X,y,test_size = 0.2, random_state = 1)


model = RandomForestClassifier(random_state=1, max_depth=16, n_estimators=250)# get instance of model
model.fit(x_train, y_train) # Train/Fit model 

y_pred = model.predict(x_test) # get y predictions
print(classification_report(y_test, y_pred)) # output accuracy


# ## Confusion Matrix

# In[71]:


from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)


# 105 are the True Positives in our test data.
# 
# There are 6 type 1 error (False Positives)- predicted positive and it’s false.
# 
# There are 6 type 2 error (False Negatives)- predicted negative and it’s false.
# 
# 121 are the True Negatives in our test data.

# In[72]:


importance = model.feature_importances_

# summarize feature importance
for i,v in enumerate(importance):
    print('Feature: %0d, Score: %.5f' % (i,v))


# In[73]:


import pandas as pd
index= heart_disease_truncated.columns[:-1]
importance = pd.Series(model.feature_importances_, index=index)
importance.nlargest(13).plot(kind='barh', colormap='winter')


# #### The top 4 significant features in the random forest model are
# 
# St slope, 
# max heart rate, 
# chest pain type, 
# cholestrol 

# # Conclusion

# Random Forest is the best model for our use case with parameters max_depth=16, n_estimators=250 on the truncated model with an accuracy score of 95%

# # Sarah's Analysis

# In[130]:


# Filtering data by POSITIVE Heart Disease patient
pos_data = heart_disease[heart_disease['target']==1]
pos_data.describe()


# In[131]:


# Filtering data by NEGATIVE Heart Disease patient
neg_data = heart_disease[heart_disease['target']==0]
neg_data.describe()


# In[132]:


print("(Positive Patients ST depression): " + str(pos_data['oldpeak'].mean()))
print("(Negative Patients ST depression): " + str(neg_data['oldpeak'].mean()))


# In[133]:


print("(Positive Patients thalach): " + str(pos_data['max heart rate'].mean()))
print("(Negative Patients thalach): " + str(neg_data['max heart rate'].mean()))


# ## Machine Learning + Predictive Analytics

# In[134]:


X = heart_disease.iloc[:, :-1].values
y = heart_disease.iloc[:, -1].values


# In[135]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X,y,test_size = 0.2, random_state = 1)


# In[136]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)


# ### Model 1: Logistic Regression

# In[137]:


from sklearn.metrics import classification_report 
from sklearn.linear_model import LogisticRegression

model1 = LogisticRegression(random_state=1) # get instance of model
model1.fit(x_train, y_train) # Train/Fit model 

y_pred1 = model1.predict(x_test) # get y predictions
print(classification_report(y_test, y_pred1)) # output accuracy


# ### Model 2: K-NN (K-Nearest Neighbors)

# In[138]:


from sklearn.metrics import classification_report 
from sklearn.neighbors import KNeighborsClassifier

model2 = KNeighborsClassifier() # get instance of model
model2.fit(x_train, y_train) # Train/Fit model 

y_pred2 = model2.predict(x_test) # get y predictions
print(classification_report(y_test, y_pred2)) # output accuracy


# ### Model 3: SVM (Support Vector Machine)

# In[139]:


from sklearn.metrics import classification_report 
from sklearn.svm import SVC

model3 = SVC(random_state=1) # get instance of model
model3.fit(x_train, y_train) # Train/Fit model 

y_pred3 = model3.predict(x_test) # get y predictions
print(classification_report(y_test, y_pred3)) # output accuracy


# ### Model 4: Naives Bayes Classifier

# In[140]:


from sklearn.metrics import classification_report 
from sklearn.naive_bayes import GaussianNB

model4 = GaussianNB() # get instance of model
model4.fit(x_train, y_train) # Train/Fit model 

y_pred4 = model4.predict(x_test) # get y predictions
print(classification_report(y_test, y_pred4)) # output accuracy


# ### Model 5: Decision Trees

# In[141]:


from sklearn.metrics import classification_report 
from sklearn.tree import DecisionTreeClassifier

model5 = DecisionTreeClassifier(random_state=1) # get instance of model
model5.fit(x_train, y_train) # Train/Fit model 

y_pred5 = model5.predict(x_test) # get y predictions
print(classification_report(y_test, y_pred5)) # output accuracy


# ### Model 6: Random Forest

# In[142]:


from sklearn.metrics import classification_report 
from sklearn.ensemble import RandomForestClassifier

model6 = RandomForestClassifier(random_state=1)# get instance of model
model6.fit(x_train, y_train) # Train/Fit model 

y_pred6 = model6.predict(x_test) # get y predictions
print(classification_report(y_test, y_pred6)) # output accuracy


# ## This for dataset without the 2 features
# 
# 

# In[143]:


X1 = heart_disease_without2f.iloc[:, :-1].values
y1 = heart_disease_without2f.iloc[:, -1].values


# In[144]:


# this for dataset without the 2 features

from sklearn.model_selection import train_test_split
x1_train, x1_test, y1_train, y1_test = train_test_split(X1,y1,test_size = 0.2, random_state = 1)


# In[145]:


# this for dataset without the 2 features
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x1_train = sc.fit_transform(x1_train)
x1_test = sc.transform(x1_test)


# ### Model 1: Logistic Regression

# In[146]:


from sklearn.metrics import classification_report 
from sklearn.linear_model import LogisticRegression

model1 = LogisticRegression(random_state=1) # get instance of model
model1.fit(x1_train, y1_train) # Train/Fit model 

y1_pred1 = model1.predict(x1_test) # get y predictions
print(classification_report(y1_test, y1_pred1)) # output accuracy


# ### Model 2: K-NN (K-Nearest Neighbors)

# In[147]:


from sklearn.metrics import classification_report 
from sklearn.neighbors import KNeighborsClassifier

model2 = KNeighborsClassifier() # get instance of model
model2.fit(x1_train, y1_train) # Train/Fit model 

y1_pred2 = model2.predict(x1_test) # get y predictions
print(classification_report(y1_test, y1_pred2)) # output accuracy


# ### Model 3: SVM (Support Vector Machine)

# In[148]:


from sklearn.metrics import classification_report 
from sklearn.svm import SVC

model3 = SVC(random_state=1) # get instance of model
model3.fit(x1_train, y1_train) # Train/Fit model 

y1_pred3 = model3.predict(x1_test) # get y predictions
print(classification_report(y1_test, y1_pred3)) # output accuracy


# ### Model 4: Naives Bayes Classifier

# In[149]:


from sklearn.metrics import classification_report 
from sklearn.naive_bayes import GaussianNB

model4 = GaussianNB() # get instance of model
model4.fit(x1_train, y1_train) # Train/Fit model 

y1_pred4 = model4.predict(x1_test) # get y predictions
print(classification_report(y1_test, y1_pred4)) # output accuracy


# ### Model 5: Decision Trees

# In[150]:


from sklearn.metrics import classification_report 
from sklearn.tree import DecisionTreeClassifier

model5 = DecisionTreeClassifier(random_state=1) # get instance of model
model5.fit(x1_train, y1_train) # Train/Fit model 

y1_pred5 = model5.predict(x1_test) # get y predictions
print(classification_report(y1_test, y1_pred5)) # output accuracy


# ### Model 6: Random Forest

# In[151]:


from sklearn.metrics import classification_report 
from sklearn.ensemble import RandomForestClassifier

model6 = RandomForestClassifier(random_state=1)# get instance of model
model6.fit(x1_train, y1_train) # Train/Fit model 

y1_pred6 = model6.predict(x1_test) # get y predictions
print(classification_report(y1_test, y1_pred6)) # output accuracy


# ### Confusion Matrix

# In[152]:


# cnfusion matrix for the random forest model < dataset without 2 features
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y1_test, y1_pred6)
print(cm)
accuracy_score(y1_test, y1_pred6)


# ##  Confusion Matrix for random Forest
# 105 the TP in our test data 
# 
# 6 &5 the number of errors
# 
# There are 6 type 1 error (False Positives)-  predicted positive and it’s false.
# 
# There are 5 type 2 error (False Negatives)-  predicted negative and it’s false.
# 
# 122 the TN in our test data 
# 
# 

# In[153]:


importance = model6.feature_importances_

# summarize feature importance
for i,v in enumerate(importance):
    print('Feature: %0d, Score: %.5f' % (i,v))


# In[154]:


index= heart_disease_without2f.columns[:-1]
importance = pd.Series(model6.feature_importances_, index=index)
importance.nlargest(13).plot(kind='barh', colormap='winter')


# #### The top 4 significant features in the random forest model are
# 
# St slope, 
# max heart rate, 
# chest pain type, 
# cholestrol 

# In[ ]:




