#!/usr/bin/env python
# coding: utf-8

# # MA 981-7-Dissertation : Banking Churn Analysis & Modelling

# In[79]:


# import libraries  
import pandas as pd              # data cleaning, reshaping, merging, etc.
import numpy as np               # mathematical functions, array operations, linear algebra routines, etc.

import seaborn as sns            # creating visually appealing statistical graphics
sns.set(style="darkgrid",font_scale=1)  # for sets the plot style to a dark grid & sets the font size of the plot to 1.5 times the default size.

import matplotlib.pyplot as plt  # for interatcive visualizations
plt.style.use("ggplot")

from scipy import stats          # for statistical analysis, like calculating statistical distributions, performing hypothesis tests, and generating random numbers.

from sklearn.metrics import accuracy_score, precision_score, recall_score ,f1_score
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import RandomForestClassifier

import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV

from imblearn.over_sampling import SMOTE  # imports the SMOTE class from imblearn.over_sampling, allowing for oversampling of the minority class in a dataset to address class imbalance issues.

import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)   # for managing warnings during program execution.


# # Loading Raw Dataset

# In[80]:


# to read raw dataset .csv file
bank_data = pd.read_csv('C:/Users/Mantosh.Nandy/OneDrive/Documents/MA981_Dissertation/Churn_Modelling.csv')  
bank_data.head()  


# # Basic understanding of Data of this dataset

# In[81]:


print(bank_data.shape)  # print the shape (number of rows & columns) of the dataFrame 'bank_data'


# In[82]:


bank_data.info()       # overview of the DataFrame's structure and characteristics for data exploration and analysis.


# In[83]:


des_data=bank_data.copy()
des_data.describe()


# In[84]:


# Checking Total No. of Missing Values.
pd.DataFrame(bank_data.isnull().sum(), columns=["Total No. of Missing Values"])


# In[85]:


# Heatmap analysis for re-checking missing values
sns.heatmap(bank_data.isnull(),cbar=False,cmap="viridis")   


# In[86]:


# Checking for duplicates values by the using of query methods
data_duplicates = bank_data[bank_data.duplicated()]
data_duplicates.head()


# In[87]:


#For descriptive Statistical analysis
target_objects = bank_data.describe(include="object").T
print(target_objects)


# In[88]:


# Taking Random sample from this dataset
bank_data.sample(10)


# In[89]:


# drop insignificant features columns from dataset
bank_data.drop(["RowNumber","CustomerId","Surname"],inplace=True,axis=1)
bank_data.info()


# In[90]:


# For Better Analysis Renaming Target variable name & its value
bank_data.columns = ['Churned' if col == 'Exited' else col for col in bank_data.columns]
bank_data["Churned"] = np.where(bank_data["Churned"] == 1, "Yes", "No")
bank_data.head()


# # Exploratory Data Analysis

# In[91]:


# Visualizing Target Variable

# Count and plot the distribution of "Churned" column
plt.figure(figsize=(12,6))
plt.subplot(1,2,1)
sns.set_palette("Set1")
axis=sns.countplot(x="Churned", data=bank_data)
axis.bar_label(axis.containers[0],fontweight="black",size=10)
plt.title("Customer Churned Distribution", fontweight="black", size=15, pad=15)

# Create a pie chart
plt.subplot(1, 2, 2)
plt.pie(bank_data["Churned"].value_counts(), labels=bank_data["Churned"].unique(), autopct="%1.1f%%", 
        colors=sns.set_palette("Set1"), textprops={"fontweight": "black"}, explode=[0, 0.2])

plt.title("Customer Churned Distribution", fontweight="black", size=15, pad=15)
plt.show()


# In[92]:


# Distribution Analysis
features = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary']

fig, axs = plt.subplots(2, 3, figsize=(20, 10))
for ax, feature in zip(axs.flatten(), features):
    sns.histplot(bank_data[feature],color="orange", kde=True, ax=ax)
    ax.set_title(f'Distribution of {feature}', fontweight="black", size=15, pad=15)
plt.tight_layout()
plt.show()


# In[93]:


# Visualizing Bank Customer Churned by Gender.

def countplot(column):
    plt.figure(figsize=(15, 5))
    axis=sns.countplot(x=column, data=bank_data, hue="Churned", palette="Set1")
    for p in axis.patches:
        plt.annotate(f"{p.get_height()/len(bank_data[column]):.1%}",
                     (p.get_x() + p.get_width() / 2, p.get_y() + p.get_height()), ha="center", fontsize=15)
    plt.title(f"Customer Churned by {column}", fontsize=15, fontweight="black", pad=15)
    plt.show()

countplot("Gender")


# In[94]:


# Visualizing Customer Churned by "Geoprahical" Region

countplot("Geography")


# In[95]:


# Visualizing Customer Churn by "HasCrCard"

countplot("HasCrCard")


# In[96]:


# Visualizing Customer Churned by "NumOfProducts"

countplot("NumOfProducts")


# In[97]:


# Visualizing Customer Churned by "IsActiveMember"

countplot("IsActiveMember")


# In[98]:


# Visualizing Customer Churned by "Tenure"

countplot("Tenure")


# In[99]:


# Relationship Analysis by Boxplot

plt.figure(figsize=(20, 10))
for i, feature in enumerate(features):
    plt.subplot(2, 3, i+1)
    sns.boxplot(data=bank_data, x='Churned', y=feature)
    plt.title(f'Boxplot of {feature} v/s Churned' , fontsize=14, fontweight="black", pad=14)
plt.tight_layout()
plt.show()


# In[100]:


# Visualising by using Violin Plots
plt.figure(figsize=(20, 10))
for i, feature in enumerate(features):
    plt.subplot(2, 3, i+1)
    sns.violinplot(x='Churned', y=feature, data=bank_data)
    plt.title(f'Violin Plot of {feature} v/s Churned' , fontweight="black", fontsize=14, pad=14)
plt.tight_layout()
plt.show()


# In[101]:


# Correlation clustering with Heatmap Analysis

# Convert String to numeric using encoding
data_encoded = pd.get_dummies(bank_data, columns=['Churned','Geography','Gender']) 
corr_matrix = data_encoded.corr()

plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap='inferno', linewidths=0.5)
plt.title('Correlation Matrix Heatmap')
plt.show()
                         


# # Feature Engineering  

# In[102]:


state = [(bank_data["NumOfProducts"]==1), (bank_data["NumOfProducts"]==2), (bank_data["NumOfProducts"]>2)]
values =     ["Single Product","Two Products","More Than 2 Products"]
bank_data["Total_Products"] = np.select(state,values)
bank_data.drop(columns="NumOfProducts", inplace=True)


# In[103]:


# Visualizing this New Feature "Total_Products"
countplot("Total_Products")


# In[104]:


# Creating New Feature From "Balance" Feature

state = [(bank_data["Balance"]==0), (bank_data["Balance"]>0)]
values = ["Nil Balance","More Than zero Balance"]
bank_data["Account_Balance"] = np.select(state, values)
bank_data.drop(columns="Balance",inplace=True)


# In[105]:


# Visualizing this New Feature "Account_Balance"
countplot("Account_Balance")


# # Data Preprocessing

# In[106]:


# Finding Unique Values of Categorical Columns

categorical_cols = ["Geography","Gender","Total_Products","Account_Balance"]

for column in categorical_cols:
    print(f"Unique Values in {column} column is:",bank_data[column].unique())
    print("-"*100,"\n")


# In[107]:


# Performing One Hot Encoding on "Categorical Features"

bank_data = pd.get_dummies(columns=categorical_cols, data=bank_data)


# In[108]:


# Encoding Target Variable

bank_data["Churned"].replace({"No":0,"Yes":1},inplace=True)
bank_data.head()


# In[109]:


# Checking Skewness of Continous Features.

columns = ["CreditScore","Age","EstimatedSalary"]
bank_data[columns].skew().to_frame().rename(columns={0:"Feature Skewness"})


# In[110]:


# Performing Log Transformation on "Age" Column.

old_age = bank_data["Age"]              ## Storing the Old Age values to compare these new values with the transformed values.
bank_data["Age"] = np.log(bank_data["Age"])


# In[111]:


# Visualizing "Age" Before and After Transformation.

plt.figure(figsize=(15,8))
plt.subplot(1,2,1)
sns.histplot(old_age, color="green", kde=True)
plt.title("Age Distribution Before Transformation", fontweight="black", size=15, pad=20)

plt.subplot(1,2,2)
sns.histplot(bank_data["Age"], color="green", kde=True)
plt.title("Age Distribution After Transformation", fontweight="black", size=15, pad=20)

plt.tight_layout()
plt.show()


# In[112]:


# Differentiate between "Features" & "Labels" for Model Training

X = bank_data.drop(columns=["Churned"])
y = bank_data["Churned"]


# # Splitting Data For the Model Training & Testing

# In[113]:


x_bank_train,x_bank_test,y_bank_train,y_bank_test = train_test_split(X,y,test_size=0.25,random_state=0)

print("Shape of x_bank_train is:",x_bank_train.shape,"\n")
print("Shape of x_bank_test is: ",x_bank_test.shape,"\n")

print("Shape of y_bank_train is:",y_bank_train.shape,"\n")
print("Shape of y_bank_test is: ",y_bank_test.shape,"\n")


# In[114]:


# To Overcome the Class-Imbalance in Target Variable - Applying SMOTE

bank_smt = SMOTE(random_state=42)
x_trn_resampled,y_trn_resampled = bank_smt.fit_resample(x_bank_train,y_bank_train)

print(x_trn_resampled.shape ,y_trn_resampled.shape)
y_trn_resampled.value_counts().to_frame()


# # Create Model by using of Decision Tree

# In[115]:


# To find the Best Parameters for the Model- Performing Grid-Search Method with cross-validation

dec_tree = DecisionTreeClassifier()

param_grid = {"max_depth":[3,4,5,6,7,8,9,10],
              "min_samples_split":[2,3,4,5,6,7,8],
              "min_samples_leaf":[1,2,3,4,5,6,7,8],
              "criterion":["gini","entropy"],
              "splitter":["best","random"],
              "max_features":["auto",None],
              "random_state":[0,42]}

grid_search = GridSearchCV(dec_tree, param_grid, cv=5, n_jobs=-1)

grid_search.fit(x_trn_resampled,y_trn_resampled)


# In[116]:


# Retrieving the Best Parameters for DecisionTree Model

best_params = grid_search.best_params_

print("Best Parameters for DecisionTree Model is:\n")
best_params


# In[117]:


# Create DecisionTree Model Using Best Parameters

dec_tree = DecisionTreeClassifier(**best_params)

dec_tree.fit(x_trn_resampled,y_trn_resampled)


# In[118]:


# Computing this Model Accuracy

y_trn_pred = dec_tree.predict(x_trn_resampled)
y_tst_pred = dec_tree.predict(x_bank_test)

print("Accuracy Score of Model`s Training Data is = ",round(accuracy_score(y_trn_resampled,y_trn_pred)*100,2),"%")
print("Accuracy Score of Model`s Testing Data  is = ",round(accuracy_score(y_bank_test,y_tst_pred)*100,2),"%")


# In[119]:


# Model Evaluation using Different Metric Values

print("Model`s F1 Score is = ",f1_score(y_bank_test,y_tst_pred,average="micro"))
print("Model`s Recall Score is = ",recall_score(y_bank_test,y_tst_pred,average="micro"))
print("Model`s Precision Score is = ",precision_score(y_bank_test,y_tst_pred,average="micro"))


# In[120]:


# Finding Importance of Features in DecisionTree Classifier

imp_data = pd.DataFrame({"Features Name":x_bank_train.columns,
                       "Importance":dec_tree.feature_importances_})
cols_features = imp_data.sort_values(by="Importance",ascending=False)

sns.set(style="whitegrid", font_scale=1.5)
plt.figure(figsize=(12,7))
sns.barplot(x="Importance", y="Features Name", data=cols_features, palette="plasma")
plt.title("Feature Importance in the Model Prediction", fontweight="black", size=15, pad=15)
plt.yticks(size=12)
plt.show()


# In[121]:


# Using Confusion Matrix performing Model Evaluation 

conf_matrix = confusion_matrix(y_bank_test,y_tst_pred)

plt.figure(figsize=(15,6))

sns.heatmap(data=conf_matrix, linewidth=5, annot=True, fmt="g", cmap="Set2")

plt.title("Using Confusion Matrix performing Model Evaluation",fontsize=15,pad=15,fontweight="black")
plt.ylabel("Actual Labels")
plt.xlabel("Predicted Labels")

plt.show()


# In[122]:


# Model Evaluation: ROC Curve and AUC curve Area Under the Curve

y_pred_proba = dec_tree.predict_proba(x_bank_test)[:][:,1]

df_actual_predict = pd.concat([pd.DataFrame(np.array(y_bank_test), columns=["y_actual"])])
df_actual_predict.index = y_bank_test.index


fpr, tpr, thresholds = roc_curve(df_actual_predict["y_actual"], y_pred_proba)
auc = roc_auc_score(df_actual_predict["y_actual"], y_pred_proba)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"AUC = {auc:.2f}",color="red")
plt.plot([0, 1], [0, 1], linestyle="-.", color="blue")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve",pad=15,fontweight="black")
plt.legend()
plt.show()


# # Create Model by using of Random Forest

# In[45]:


rfm = RandomForestClassifier()

param_grid = {"max_depth":[3,4,5,6,7,8],
              "min_samples_split":[3,4,5,6,7,8],
              "min_samples_leaf":[3,4,5,6,7,8],
              "n_estimators": [50,70,90,100],
              "criterion":["gini","entropy"]}

grid_search = GridSearchCV(rfm, param_grid, cv=5, n_jobs=-1)

grid_search.fit(x_trn_resampled,y_trn_resampled)


# In[46]:


# Retrieving the Best Parameters for Randomforest Model

best_params = grid_search.best_params_

print("Best Parameters for RandomForest Model is:\n\n")
best_params


# In[47]:


# Create RandmForest Model Using Best Parameters

rfm = RandomForestClassifier(**best_params)

rfm.fit(x_trn_resampled,y_trn_resampled)


# In[48]:


# Computing this Model Accuracy

y_trn_pred = rfm.predict(x_trn_resampled)
y_tst_pred = rfm.predict(x_bank_test)

print("Accuracy Score of Model`s Training Data is = ",round(accuracy_score(y_trn_resampled,y_trn_pred)*100,2),"%")
print("Accuracy Score of Model`s Testing Data  is = ",round(accuracy_score(y_bank_test,y_tst_pred)*100,2),"%")


# In[49]:


# Model Evaluation using Different Metric Values

print("Model`s F1 Score is = ",f1_score(y_bank_test,y_tst_pred,average="micro"))
print("Model`s Recall Score is = ",recall_score(y_bank_test,y_tst_pred,average="micro"))
print("Model`s Precision Score is = ",precision_score(y_bank_test,y_tst_pred,average="micro"))


# In[50]:


# Finding Importance of Features in RandomForest Model

imp_df = pd.DataFrame({"Feature Name":x_bank_train.columns,
                       "Importance":rfm.feature_importances_})
features = imp_df.sort_values(by="Importance",ascending=False)

plt.figure(figsize=(12,7))
sns.barplot(x="Importance", y="Feature Name", data=features, palette="plasma")
plt.title("Feature Importance in the Model Prediction", fontweight="black", size=20, pad=20)
plt.yticks(size=12)
plt.show()


# In[51]:


# Using Confusion Matrix performing Model Evaluation 

conf_matrix = confusion_matrix(y_bank_test,y_tst_pred)

plt.figure(figsize=(15,6))

sns.heatmap(data=conf_matrix, linewidth=5, annot=True, fmt="g", cmap="Set2")

plt.title("Using Confusion Matrix performing Model Evaluation",fontsize=15,pad=15,fontweight="black")
plt.ylabel("Actual Labels")
plt.xlabel("Predicted Labels")

plt.show()


# In[52]:


# Model Evaluation: ROC Curve and AUC curve Area Under the Curve

y_pred_proba = rfm.predict_proba(x_bank_test)[:][:,1]

df_actual_predict = pd.concat([pd.DataFrame(np.array(y_bank_test), columns=["y_actual"])])
df_actual_predict.index = y_bank_test.index


fpr, tpr, thresholds = roc_curve(df_actual_predict["y_actual"], y_pred_proba)
auc = roc_auc_score(df_actual_predict["y_actual"], y_pred_proba)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"AUC = {auc:.2f}",color="red")
plt.plot([0, 1], [0, 1], linestyle="-.", color="blue")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve",fontsize=15,fontweight="black")
plt.legend()
plt.show()


# # Create Model by using of XGBoost Algorithm

# In[53]:


xgb_clf = xgb.XGBClassifier()

param_grid = {"max_depth":[3,4,5,6,7,8],
              "learning_rate": [0.1, 0.01, 0.05],
              "gamma": [0, 0.1, 0.2],
              "n_estimators": [50,70,90,100],
              "min_child_weight": [1, 2, 3],
              "subsample": [0.6, 0.8, 1],
               "colsample_bytree": [0.6, 0.8, 1]}

grid_search = GridSearchCV(xgb_clf, param_grid, cv=5, n_jobs=-1)

grid_search.fit(x_trn_resampled,y_trn_resampled)


# In[54]:


# Retrieving the Best Parameters for XGBoost Model

best_params = grid_search.best_params_

print("Best Parameters for DecisionTree Model is:\n")
best_params


# In[55]:


# Create XGboost Model Using Best Parameters

xgb = XGBClassifier(**best_params)

xgb.fit(x_trn_resampled,y_trn_resampled)


# In[56]:


# Computing this Model Accuracy

y_trn_pred = xgb.predict(x_trn_resampled)
y_tst_pred = xgb.predict(x_bank_test)

print("Accuracy Score of Model`s Training Data is = ",round(accuracy_score(y_trn_resampled,y_trn_pred)*100,2),"%")
print("Accuracy Score of Model`s Testing Data  is = ",round(accuracy_score(y_bank_test,y_tst_pred)*100,2),"%")


# In[57]:


# Model Evaluation using Different Metric Values

print("Model`s F1 Score is = ",f1_score(y_bank_test,y_tst_pred,average="micro"))
print("Model`s Recall Score is = ",recall_score(y_bank_test,y_tst_pred,average="micro"))
print("Model`s Precision Score is = ",precision_score(y_bank_test,y_tst_pred,average="micro"))


# In[58]:


# Finding Importance of Features in XGBoost Model

imp_df = pd.DataFrame({"Feature Name":x_bank_train.columns,
                       "Importance":xgb.feature_importances_})
features = imp_df.sort_values(by="Importance",ascending=False)

plt.figure(figsize=(12,7))
sns.barplot(x="Importance", y="Feature Name", data=features, palette="plasma")
plt.title("Feature Importance in the Model Prediction", fontweight="black", size=20, pad=20)
plt.yticks(size=12)
plt.show()


# In[59]:


# Using Confusion Matrix performing Model Evaluation 

conf_matrix = confusion_matrix(y_bank_test,y_tst_pred)

plt.figure(figsize=(15,6))

sns.heatmap(data=conf_matrix, linewidth=5, annot=True, fmt="g", cmap="Set2")

plt.title("Using Confusion Matrix performing Model Evaluation",fontsize=15,pad=15,fontweight="black")
plt.ylabel("Actual Labels")
plt.xlabel("Predicted Labels")

plt.show()


# In[60]:


# Model Evaluation: ROC Curve and AUC curve Area Under the Curve

y_pred_proba = xgb.predict_proba(x_bank_test)[:][:,1]

df_actual_predict = pd.concat([pd.DataFrame(np.array(y_bank_test), columns=["y_actual"])])
df_actual_predict.index = y_bank_test.index


fpr, tpr, thresholds = roc_curve(df_actual_predict["y_actual"], y_pred_proba)
auc = roc_auc_score(df_actual_predict["y_actual"], y_pred_proba)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"AUC = {auc:.2f}",color="red")
plt.plot([0, 1], [0, 1], linestyle="-.", color="blue")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve",fontsize=15,fontweight="black")
plt.legend()
plt.show()


# In[123]:


import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

# Fit the models

dec_tree = DecisionTreeClassifier(max_depth=10, min_samples_split=5, random_state=42,
                       splitter='random')
dec_tree.fit(x_trn_resampled,y_trn_resampled)

rfm = RandomForestClassifier(criterion='entropy', max_depth=8, min_samples_leaf=5,
                       min_samples_split=7, n_estimators=50)
rfm.fit(x_trn_resampled,y_trn_resampled)

xgb = xgb.XGBClassifier(base_score=None, booster=None, callbacks=None,
              colsample_bylevel=None, colsample_bynode=None,
              colsample_bytree=0.6, device=None, early_stopping_rounds=None,
              enable_categorical=False, eval_metric=None, feature_types=None,
              gamma=0.2, grow_policy=None, importance_type=None,
              interaction_constraints=None, learning_rate=0.05, max_bin=None,
              max_cat_threshold=None, max_cat_to_onehot=None,
              max_delta_step=None, max_depth=8, max_leaves=None,
              min_child_weight=1, monotone_constraints=None,
              multi_strategy=None, n_estimators=90, n_jobs=None,
              num_parallel_tree=None, random_state=None)
xgb.fit(x_trn_resampled,y_trn_resampled)

# Compute the ROC curves and AUC scores for each model
y_pred_dt = dec_tree.predict_proba(x_bank_test)[:, 1]
fpr_dt, tpr_dt, _ = roc_curve(y_bank_test, y_pred_dt)
roc_auc_dt = auc(fpr, tpr)

y_pred_rf = rfm.predict_proba(x_bank_test)[:, 1]
fpr_rf, tpr_rf, _ = roc_curve(y_bank_test, y_pred_rf)
roc_auc_rf = auc(fpr_rf, tpr_rf)

y_pred_xgb = xgb.predict_proba(x_bank_test)[:, 1]
fpr_xgb, tpr_xgb, _ = roc_curve(y_bank_test, y_pred_xgb)
roc_auc_xgb = auc(fpr_xgb, tpr_xgb)

# Create a figure 
plt.figure(figsize=(8, 6))

# Plot the ROC curves
plt.plot(fpr_dt, tpr_dt, color='blue', label='Decision Tree (AUC = {:.2f})'.format(roc_auc_dt))
plt.plot(fpr_rf, tpr_rf, color='red', label='Random Forest (AUC = {:.2f})'.format(roc_auc_rf))
plt.plot(fpr_xgb, tpr_xgb, color='green', label='XGBoost (AUC = {:.2f})'.format(roc_auc_xgb))
plt.plot([0, 1], [0, 1], '-.', color='black')

# Set title and labels
plt.title('ROC Curves for Different Models')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()


# In[ ]:




