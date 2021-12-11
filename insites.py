#!/usr/bin/env python
# coding: utf-8

# # Backorder Prediction

# In[1]:


# Importing the libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


# Reading the Data for train and test.
data=pd.read_csv('Kaggle_Training_Dataset_v2.csv',low_memory=False)
data1=pd.read_csv('Kaggle_Test_Dataset_v2.csv',low_memory=False)


# In[3]:


print("The training data shape =",data.shape ,". The test data shape =", data1.shape)


# In[4]:


# concating train and test data.
data=pd.concat([data,data1])
data.head()


# In[5]:


print("The final shape of the data is =",data.shape)


# In[6]:


data.columns


# In[8]:


data.describe()


# In[9]:


data['went_on_backorder'].unique()


# In[10]:


data.groupby('went_on_backorder').count()


# Out of 1929937, only 13981 i.e., 0.5% of data is Out of Stock (Flag as Yes... as it went to BackOrder), and more than 99.5% is in Stock (Flag as No). So its result bias.

# In[11]:


# checking now the nullvalues
percentage1 = data.isnull().sum()
percentage2 = data.isnull().sum()/data.isnull().count()*100
missing_values=pd.concat([percentage1,percentage2], axis=1,keys=['Total','%'])
missing_values


# # Data Cleaning

#  1. need to clean missing values.
#  2. need to drop sku column as it is random product id.
#  3. Replacce all catagorical columns with 1 for yes and 0 for no.
#  4. we need to scale the numerical columns.
#

# In[12]:


data.drop(['sku'],axis=1,inplace=True)


# In[13]:


data.head()


# In[14]:


categorical_columns = ['potential_issue', 'deck_risk', 'oe_constraint', 'ppap_risk','stop_auto_buy', 'rev_stop', 'went_on_backorder']

for col in categorical_columns:
    data[col] = data[col].map({'No':0, 'Yes':1})


# In[15]:


data.info()


# In[16]:


data.columns


# In[17]:


# filling all missing values.
data.national_inv = data.national_inv.fillna(data.national_inv.median())
data.lead_time = data.lead_time.fillna(data.lead_time.median())
data.in_transit_qty = data.in_transit_qty.fillna(data.in_transit_qty.median())
data.forecast_3_month = data.forecast_3_month.fillna(data.forecast_3_month.median())
data.forecast_6_month = data.forecast_6_month.fillna(data.forecast_6_month.median())
data.forecast_9_month = data.forecast_9_month.fillna(data.forecast_9_month.median())
data.sales_1_month = data.sales_1_month.fillna(data.sales_1_month.median())
data.sales_3_month = data.sales_3_month.fillna(data.sales_3_month.median())
data.sales_6_month = data.sales_6_month.fillna(data.sales_6_month.median())
data.sales_9_month = data.sales_9_month.fillna(data.sales_9_month.median())
data.potential_issue = data.potential_issue.fillna(data.potential_issue.median())
data.min_bank = data.min_bank.fillna(data.min_bank.median())
data.pieces_past_due = data.pieces_past_due.fillna(data.pieces_past_due.median())
data.perf_6_month_avg = data.perf_6_month_avg.fillna(data.perf_6_month_avg.median())
data.perf_12_month_avg = data.perf_12_month_avg.fillna(data.perf_12_month_avg.median())
data.local_bo_qty = data.local_bo_qty.fillna(data.local_bo_qty.median())
data.deck_risk = data.deck_risk.fillna(data.deck_risk.median())
data.oe_constraint = data.oe_constraint.fillna(data.oe_constraint.median())
data.ppap_risk = data.ppap_risk.fillna(data.ppap_risk.median())
data.stop_auto_buy = data.stop_auto_buy.fillna(data.stop_auto_buy.median())
data.rev_stop = data.rev_stop.fillna(data.rev_stop.median())
data.went_on_backorder = data.went_on_backorder.fillna(data.went_on_backorder.median())


# In[18]:


percentage1 = data.isnull().sum()
percentage2 = data.isnull().sum()/data.isnull().count()*100
missing_values=pd.concat([percentage1,percentage2], axis=1,keys=['Total','%'])
missing_values


# # Looking at correlations between features and the label

# In[19]:


import matplotlib.pyplot as plt
import seaborn as sns
data.corr().round(2)


# In[20]:


plt.figure(figsize=(20,20))
cor = data.corr()
sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
plt.show()


# The correlation matrix shows that the in_transit_qty, forecast_3_month, forecast_6_month, forecast_9_month, sales_1_month, sales_3_month, sales_6_month, sales_9_month, and min_bank are highly correlated

# In[21]:


# Dropping some features.
features = ['national_inv', 'lead_time', 'sales_1_month', 'pieces_past_due', 'perf_6_month_avg',
            'local_bo_qty', 'deck_risk', 'oe_constraint', 'ppap_risk', 'stop_auto_buy', 'rev_stop']


# In[22]:


X = data[features]
y = data['went_on_backorder']


# In[23]:


display(X.shape, y.shape)


# # Change scale of data

# In[24]:


from sklearn.preprocessing import MinMaxScaler


# In[25]:


scaler = MinMaxScaler()
scaler.fit(X)

X = scaler.transform(X)
X = pd.DataFrame(X, columns=features)


# In[26]:


X.head()


# # Near Miss Undersampling

# In[34]:


# define the undersampling method
from imblearn.under_sampling import NearMiss
undersample = NearMiss(version=1, n_neighbors=3)


# In[35]:


# transform the dataset
X, y = undersample.fit_resample(X, y)


# In[36]:


X.head()


# In[37]:


y.tail()


# # Splitting the Dataset

# In[38]:


from sklearn.model_selection import train_test_split
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2)

print (X_train.shape, y_train.shape)
print (X_valid.shape, y_valid.shape)


# # 1. RandomForstClassifier

# In[39]:


from sklearn.ensemble import RandomForestClassifier


# In[40]:


random_forest = RandomForestClassifier(n_estimators=10)
random_forest.fit(X_train, y_train)
Y_pred_rf = random_forest.predict(X_valid)
acc_random_forest = round(random_forest.score(X_train, y_train) * 100, 2)


# In[41]:


acc_random_forest


# # 2. DecisionTreeClassifier

# In[42]:


from sklearn.tree import DecisionTreeClassifier


# In[43]:


decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, y_train)
Y_pred_dt = decision_tree.predict(X_valid)
acc_decision_tree = round(decision_tree.score(X_train, y_train) * 100, 2)


# In[44]:


acc_decision_tree


# # 3. LogisticRegression

# In[45]:


from sklearn.linear_model import LogisticRegression


# In[46]:


logreg = LogisticRegression()
logreg.fit(X_train, y_train)
Y_valid_pred_lr = logreg.predict(X_valid)
acc_LogisticRegression  = round(logreg.score(X_train, y_train) * 100, 2)


# In[47]:


acc_LogisticRegression


# In[48]:


modelling_score = pd.DataFrame({
    'Model': ['Linear Regression','Random Forest','Decision Tree'],
    'Score': [acc_LogisticRegression, acc_random_forest, acc_decision_tree]})


# In[49]:


modelling_score


# Thanks

# #  Doing Hyper parameter Tuning

# In[50]:


import numpy as np
from sklearn.model_selection import RandomizedSearchCV
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 100, stop = 2000, num = 25)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 50, num = 10)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
print(random_grid)


# In[51]:


rf = RandomForestClassifier()
# Random search of parameters, using 2 fold cross validation,
# search across 50 different combinations.
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter =50, cv = 2, verbose=2, random_state=7, n_jobs = -1)
rf_random.fit(X_train, y_train)
rf_random.best_estimator_


# In[72]:


rdf_clf = RandomForestClassifier(bootstrap=False, ccp_alpha=0.0, class_weight=None,
                        max_depth=36, max_features='sqrt',
                       max_leaf_nodes=None, max_samples=2,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=5,
                       min_weight_fraction_leaf=0.0, n_estimators=495,
                       n_jobs=None, oob_score=False, random_state=7,
                       verbose=3, warm_start=False)

rdf_clf.fit(X_train, y_train)


# In[73]:


Y_pred_rf=rdf_clf.predict(X_valid)


# In[74]:


acc_random_forest = round(rdf_clf.score(X_train, y_train) * 100, 2)


# In[75]:


print(acc_random_forest)


# #  Checking For yes.

# In[64]:


test=[[0.002206,1.000000,0.000065,0.0,0.0000,0.00008,1.0,0.0,1.0,0.0,0.0]]


# In[65]:


answer=rdf_clf.predict(test)
print(answer)


# # Checking For NO

# In[66]:


test1=[[0.002205,0.153846,0.000000,0.0,0.0000,0.0,0.0,0.0,0.0,1.0,0.0]]


# In[67]:


answer=rdf_clf.predict(test1)
print(answer)


# In[ ]: