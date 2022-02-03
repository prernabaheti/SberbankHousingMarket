#!/usr/bin/env python
# coding: utf-8

# #***Sign in to Google Drive***

# In[1]:



# To upload our datasets from our working directory we need to mount our drive contents to the colab environment. 
# For the code to do so you can search “mount” in code snippets or use the code given below. 
# Our entire drive contents are now mounted on colab at the location “/gdrive”.
from google.colab import drive
drive.mount('/gdrive')
#Change current working directory to gdrive
get_ipython().run_line_magic('cd', '/gdrive')


# #***Import Necessary Packages***

# In[2]:


get_ipython().system('pip install vecstack')


# In[4]:


from vecstack import stacking
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score #works
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from imblearn.over_sampling import SMOTE 
from sklearn.svm import SVC
from collections import Counter #for Smote, 
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import cross_val_score

import warnings
warnings.filterwarnings("ignore")


# #***Import Test and Train Files from Kaggle***

# In[5]:


trainfile = r'/gdrive/My Drive/train.csv'
train_data = pd.read_csv(trainfile)

#train_data = pd.read_csv("C:/Users/admin/Downloads/Insurance Fraud - TRAIN-3000(1).csv")


testfile = r'/gdrive/My Drive/test.csv'
test_data = pd.read_csv(testfile)

#test_data = pd.read_csv("C:/Users/admin/Downloads/Insurance Fraud -TEST-12900(1).csv")


print(train_data.shape)
print(train_data.head())    


# #***DEA and Handling Missing values***

# In[6]:


train_data.isna()


# In[8]:


train_data.isna().sum()


# In[9]:


test_data.isna()


# In[10]:


test_data.isna().sum()


# In[11]:


#median value imputation
train_data= train_data.fillna(train_data.median())
test_data= test_data.fillna(test_data.median())


# In[26]:


#Identify the object columns
object_columns = list(train_data.select_dtypes(include=['object']).columns)
print(object_columns)


# In[29]:


#Drop the time stamp
train_data=train_data.drop(columns=['product_type', 'sub_area', 'culture_objects_top_25', 'thermal_power_plant_raion', 'incineration_raion', 'oil_chemistry_raion', 'radiation_raion', 'railroad_terminal_raion', 'big_market_raion', 'nuclear_reactor_raion', 'detention_facility_raion', 'water_1line', 'big_road1_1line', 'railroad_1line', 'ecology']) 
test_data= test_data.drop(columns=['product_type', 'sub_area', 'culture_objects_top_25', 'thermal_power_plant_raion', 'incineration_raion', 'oil_chemistry_raion', 'radiation_raion', 'railroad_terminal_raion', 'big_market_raion', 'nuclear_reactor_raion', 'detention_facility_raion', 'water_1line', 'big_road1_1line', 'railroad_1line', 'ecology']) 


# #*Split train data for validation*

# In[30]:


X= train_data.iloc[:, :-1].values
y= train_data['price_doc']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


# In[31]:


X_Test= test_data.copy()


# In[32]:


#Quick peek into the target variable
y.value_counts()


# #Decision Tree

# In[55]:


#Decision Tree Regressor ========================================================================
#CONSTRUCT DEFAULT DECISION TREE AND OBTAIN RESPECTIVE ACCURACY 
clf = DecisionTreeRegressor()
clf.fit(X_train, y_train)
clf_predict_Train=clf.predict(X_train)

#clf.feature_importances_
mean_squared_error(y_train,clf_predict_Train)
print("RMSE (training) for Decision Tree:{0:10f}".format(mean_squared_error(y_train,clf_predict_Train)))
clf_predict_Test=clf.predict(X_test)
mean_squared_error(y_test,clf_predict_Test)
print("RMSE (Test Data) for Decision Tree:{0:10f}".format(mean_squared_error(y_test,clf_predict_Test)))


# In[34]:


#Hyperparameter tuning done for decision tree classifier
parameters={'min_samples_split' : range(10,100,10),'max_depth': range(1,20,2)}
clf_random = RandomizedSearchCV(clf,parameters,n_iter=15)
clf_random.fit(X_train, y_train)
grid_parm=clf_random.best_params_
print(grid_parm)

#Using the parameters obtained from HyperParameterTuning in the DecisionTreeClassifier 
clf = DecisionTreeClassifier(**grid_parm)
clf.fit(X_train,y_train)
clf_predict = clf.predict(X_test)

#Obtain accuracy ,confusion matrix,classification report and AUC values for the result above.
print("accuracy Score (training) after hypertuning for Decision Tree:{0:6f}".format(clf.score(X_test,y_test)))
print("Confusion Matrix after hypertuning for Decision Tree")
print(confusion_matrix(y_test,clf_predict))
print("=== Classification Report ===")
print(classification_report(y_test,clf_predict))

#get cross-validation report
clf_cv_score = cross_val_score(clf, X_train, y_train, cv=10, scoring="roc_auc")
print("=== All AUC Scores ===")
print(clf_cv_score)
print('\n')
print("=== Mean AUC Score ===")
print("Mean AUC Score - Decision Tree: ",clf_cv_score.mean())


# In[56]:


clf_predict=clf.predict(X_Test)
clf_predict
result = pd.DataFrame(X_Test['id'])
result['price_doc']=clf_predict
result.to_csv('/gdrive/My Drive/DecisionTreeResults.csv',index=None)
result.head()


# #Random Forest

# In[36]:


#Random Forest Regressor==============================================================================
#=================================================================================================

rfc = RandomForestRegressor()
rfc.fit(X_train, y_train)
rfc_predict_Train=rfc.predict(X_train)

mean_squared_error(y_train,rfc_predict_Train)
print("RMSE (training) for Decision Tree:{0:10f}".format(mean_squared_error(y_train,rfc_predict_Train)))
rfc_predict_Test=rfc.predict(X_test)
mean_squared_error(y_test,rfc_predict_Test)
print("RMSE (Test Data) for Decision Tree:{0:10f}".format(mean_squared_error(y_test,rfc_predict_Test)))


# In[37]:


#Hyperparameter tuning for random forest classifier
rfc_random = RandomizedSearchCV(rfc,parameters,n_iter=15)
rfc_random.fit(X_train, y_train)
grid_parm_rfc=rfc_random.best_params_
print(grid_parm_rfc)

#Construct Random Forest with best parameters
rfc= RandomForestClassifier(**grid_parm_rfc)
rfc.fit(X_train,y_train)
rfc_predict = rfc.predict(X_test)
print("accuracy Score (training) after hypertuning for Random Forest:{0:6f}".format(rfc.score(X_test,y_test)))
print("Confusion Matrix after hypertuning for Random Forest:")
print(confusion_matrix(y_test,rfc_predict))
print("=== Classification Report ===")
print(classification_report(y_test,rfc_predict))

#get cross-validation report
#rfc_cv_score = cross_val_score(rfc, X_train, y_train, cv=10, scoring="roc_auc")
#print("=== All AUC Scores ===")
#print(rfc_cv_score)
#print('\n')
#print("=== Mean AUC Score ===")
#print("Mean AUC Score - Random Forest: ",rfc_cv_score.mean())


# In[38]:


rfc_predict=rfc.predict(X_Test)
rfc_predict
result = pd.DataFrame(X_Test['id'])
result.head()
result['price_doc']=rfc_predict
result.to_csv('/gdrive/My Drive/RandomForestResults.csv',index=None)


# #Gradient Boosting
# 

# In[39]:


#Gradient Boosting Regressor================================================================================

abc =GradientBoostingRegressor()
abc.fit(X_train, y_train)
abc_predict_Train=abc.predict(X_train)

mean_squared_error(y_train,abc_predict_Train)
print("RMSE (training) for Decision Tree:{0:10f}".format(mean_squared_error(y_train,abc_predict_Train)))
abc_predict_Test=rfc.predict(X_test)
mean_squared_error(y_test,abc_predict_Test)
print("RMSE (Test Data) for Decision Tree:{0:10f}".format(mean_squared_error(y_test,abc_predict_Test)))


# In[ ]:


#Construct Gradient Boosting model

search_grid={'n_estimators':[5,10,20],'learning_rate':[0.01,.1]}
abc =GradientBoostingClassifier()
abc.fit(X_train, y_train)
abc_predict=abc.predict(X_test)
print("accuracy Score (training) for Boosting:{0:6f}".format(abc.score(X_test,y_test)))
print("Confusion Matrix for boosting:")
print(confusion_matrix(y_test,abc_predict))
abc_random = RandomizedSearchCV(abc,search_grid,n_iter=15)
abc_random.fit(X_train, y_train)
grid_parm_abc=abc_random.best_params_
print(grid_parm_abc)
abc= GradientBoostingClassifier(**grid_parm_abc)
abc.fit(X_train,y_train)
abc_predict = abc.predict(X_test)
print("accuracy Score (training) after hypertuning for Boosting:{0:6f}".format(abc.score(X_test,y_test)))
print("Confusion Matrix after hypertuning for Boosting:")
print(confusion_matrix(y_test,abc_predict))
print("=== Classification Report ===")
print(classification_report(y_test,abc_predict))
abc_cv_score = cross_val_score(abc, X_train, y_train, cv=10, scoring="roc_auc")
print("=== All AUC Scores ===")
print(abc_cv_score)
print('\n')
print("=== Mean AUC Score ===")
print("Mean AUC Score - Boosting: ",abc_cv_score.mean())


# In[40]:


abc_predict=abc.predict(X_Test)
abc_predict
result = pd.DataFrame(X_Test['id'])
result.head()
result['price_doc']=abc_predict
result.to_csv('/gdrive/My Drive/GradientBoostingResults.csv',index=None)


# #***Stacking Ensemble Method***

# In[51]:


#STACKING MODELS =====================================================================
print("___________________________________________________________________________________________\nEnsemble Methods Predictions using GradientBoosting, RandomForest and Decision Tree Classifier\n")

models = [ GradientBoostingRegressor(), RandomForestRegressor(), DecisionTreeRegressor() ]
      
S_Train, S_Test = stacking(models,                   
                           X_train, y_train, X_Test,   
                           regression=True, 
     
                           mode='oof_pred_bag', 
       
                           needs_proba=False,
         
                           save_dir=None, 
                                        
                           n_folds=4, 
                                                    
                           verbose=2)


# In[52]:


model = GradientBoostingRegressor()
    
model = model.fit(S_Train, y_train)
y_pred = model.predict(S_Test)
#print('Final prediction score for ensemble methods: [%.8f]' % accuracy_score(y_test, y_pred))


# In[ ]:


search_grid={'n_estimators':[5,10,20],'learning_rate':[0.01,.1]}
model_random = RandomizedSearchCV(model,search_grid,n_iter=5)
model_random.fit(X_train, y_train)
grid_parm_model=model_random.best_params_
print(grid_parm_model)
model= GradientBoostingClassifier(**grid_parm_model)
model.fit(X_train,y_train)
model_predict = model.predict(X_test)
print("accuracy Score (training) after hypertuning for Boosting:{0:6f}".format(model.score(X_test,y_test)))
print("Confusion Matrix after hypertuning for Boosting:")
print(confusion_matrix(y_test,model_predict))
print("=== Classification Report ===")
print(classification_report(y_test,model_predict))
model_cv_score = cross_val_score(model, X_train, y_train, cv=10, scoring="roc_auc")
print("=== All AUC Scores ===")
print(model_cv_score)
print('\n')
print("=== Mean AUC Score ===")
print("Mean AUC Score - Boosting: ",model_cv_score.mean())


# In[ ]:


mean_squared_error(y_train,y_pred_train)
print("RMSE (training) for Decision Tree:{0:10f}".format(mean_squared_error(y_train,y_pred_train)))
mean_squared_error(y_test,y_pred_test)
print("RMSE (Test Data) for Decision Tree:{0:10f}".format(mean_squared_error(y_test,y_pred_test)))


# In[54]:


#Get Prediction Probability for the predicted class as a dataframe
#pred_Probability =pd.DataFrame(model.predict_proba(S_Test))
result = pd.DataFrame(X_Test['id'])
result.head()
result['price_doc'] = y_pred
#pred_Probability.head()
result.to_csv('/gdrive/My Drive/StackedModelResults.csv',index=None)


# In[48]:


#STACKING - CONTRUCT A GRADIENT BOOSTING MODEL==============================
model = GradientBoostingRegressor()
    
model = model.fit(S_Train, y_train)
y_pred_train = model.predict(S_Train)
y_pred_test = model.predict(S_Test)


# In[45]:


y_pred_train


# In[46]:


y_pred_test

