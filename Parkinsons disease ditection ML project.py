#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Required Libraries
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


# Specify the file path
file_path = r"D:\CV things\ML projects\parkinsons.data"

# Read the CSV file into a DataFrame
df = pd.read_csv(file_path)

# Displaying the first five rows as an example
df


# In[3]:


#Matrix column entries (attributes):
#name - ASCII subject name and recording number
#MDVP:Fo(Hz) - Average vocal fundamental frequency
#MDVP:Fhi(Hz) - Maximum vocal fundamental frequency
#MDVP:Flo(Hz) - Minimum vocal fundamental frequency
#MDVP:Jitter(%),MDVP:Jitter(Abs),MDVP:RAP,MDVP:PPQ,Jitter:DDP - Several 
#measures of variation in fundamental frequency
#MDVP:Shimmer,MDVP:Shimmer(dB),Shimmer:APQ3,Shimmer:APQ5,MDVP:APQ,Shimmer:DDA - Several measures of variation in amplitude
#NHR,HNR - Two measures of ratio of noise to tonal components in the voice
#status - Health status of the subject (one) - Parkinson's, (zero) - healthy
#RPDE,D2 - Two nonlinear dynamical complexity measures
#DFA - Signal fractal scaling exponent
#spread1,spread2,PPE - Three nonlinear measures of fundamental frequency variation


# In[4]:


# Drop 'name' column
df.drop(['name'], axis=1, inplace=True)

# Display DataFrame after dropping 'name' column
df.head()


# In[5]:


# Visualize distribution of 'MDVP:Fo(Hz)'
plt.figure(figsize=(6, 4))
sns.histplot(df['MDVP:Fo(Hz)'], bins=20, kde=True)
plt.title('Distribution of MDVP:Fo(Hz)')
plt.xlabel('MDVP:Fo(Hz)')
plt.ylabel('Frequency')
plt.show()


# In[6]:


# Define X (features) and Y (target)
X = df.drop('status', axis=1)
Y = df['status']

# Split data into train and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=40)


# In[7]:


X.head()


# In[8]:


Y.head()


# In[9]:


# Visualize distribution of target class (status)
plt.figure(figsize=(6, 4))
sns.countplot(x='status', data=df)
plt.title('Distribution of Target Class (status)')
plt.xlabel('Status')
plt.ylabel('Count')
plt.show()


# In[10]:


# Initialize and fit a Random Forest model
RF = RandomForestClassifier(random_state=40)
RF.fit(X_train, Y_train)

# Predict on test data
test_preds_rf = RF.predict(X_test)

# Model evaluation after hyperparameter tuning
print("Model accuracy on test data (Before Hyperparameter Tuning):", accuracy_score(Y_test, test_preds_rf))
print("Confusion matrix:\n", confusion_matrix(Y_test, test_preds_rf))
print("Classification Report:\n", classification_report(Y_test, test_preds_rf))


# In[11]:


# Hyperparameter Tuning
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 5, 10, 15],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}


RF = RandomForestClassifier(random_state=40)
grid_search = GridSearchCV(estimator=RF, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, Y_train)


# In[12]:


# Get the best parameters
best_params = grid_search.best_params_

print("Best parameters found:", best_params)

# Initialize and fit a Random Forest model with optimized hyperparameters
best_RF = RandomForestClassifier(random_state=40, **best_params)
best_RF.fit(X_train, Y_train)

# Predict on test data
test_preds_best_RF = best_RF.predict(X_test)


# In[13]:


# Perform cross-validation
cv_scores = cross_val_score(best_RF, X, Y, cv=5)
print("Cross-validation scores:", cv_scores)
print("Mean CV accuracy:", cv_scores.mean())


# In[14]:


# Model evaluation after hyperparameter tuning
print("Model accuracy on test data (After Hyperparameter Tuning):", accuracy_score(Y_test, test_preds_best_RF))
print("Confusion matrix:\n", confusion_matrix(Y_test, test_preds_rf))
print("Classification Report:\n", classification_report(Y_test, test_preds_rf))


# In[15]:


# Feature Importance
feature_importances = best_RF.feature_importances_

# Create DataFrame for feature importance
feature_importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importances})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)


# In[16]:


# Update table after feature importance calculation
print("\nTable after Feature Importance calculation:")
feature_importance_df


# In[17]:


# Visualize feature importance
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
plt.title('Feature Importance')
plt.xlabel('Importance Score')
plt.ylabel('Features')
plt.show()


# In[18]:


# Select top features based on importance scores (e.g., top 7 features)
top_features = feature_importance_df['Feature'][:7].tolist()

# Create new X with selected top features
X_top_features = df[top_features]

# Splitting data with the top selected features
X_train_top, X_test_top, Y_train, Y_test = train_test_split(X_top_features, Y, test_size=0.2, random_state=40)


# In[19]:


# Create a new DataFrame for feature importance with the top 7 features
top_features_df = feature_importance_df.head(7)

# Visualize feature importance using bar plot
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=top_features_df, palette='viridis')
plt.title('Top 7 Feature Importance')
plt.xlabel('Importance Score')
plt.ylabel('Features')
plt.show()


# In[20]:


# Create a pivot table for the feature importance data
pivot_df = top_features_df.pivot(index=None, columns='Feature', values='Importance').fillna(0)

# Create the heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(pivot_df, cmap='viridis', annot=True, fmt='.3f')
plt.title('Feature Importance Heatmap')
plt.show()


# In[21]:


# Initializing and fitting the Random Forest model with the selected features
best_params = {'n_estimators': 200, 'max_depth': None, 'min_samples_leaf': 1, 'min_samples_split': 2}
best_RF_selected = RandomForestClassifier(random_state=40, **best_params)
best_RF_selected.fit(X_train_top, Y_train)

# Predict on test data with selected features
test_preds_rf_selected = best_RF_selected.predict(X_test_top)


# In[22]:


#Model evaluation with selected features
print("\nModel accuracy on test data (After Feature Selection):", accuracy_score(Y_test, test_preds_rf_selected))
print("Confusion matrix:\n", confusion_matrix(Y_test, test_preds_rf_selected))
print("Classification Report:\n", classification_report(Y_test, test_preds_rf_selected))


# In[23]:


acc_before_feature_selection = accuracy_score(Y_test, test_preds_rf)*100
acc_after_feature_selection = accuracy_score(Y_test, test_preds_rf_selected)*100

# Assuming acc_before_feature_selection and acc_after_feature_selection are accuracy scores
accuracy_scores = [acc_before_feature_selection, acc_after_feature_selection]
labels = ['Before Selection', 'After Selection']

plt.figure(figsize=(6, 4))
sns.barplot(x=labels, y=accuracy_scores)
plt.title('Model Accuracy Before and After Feature Selection')
plt.xlabel('Model')
plt.ylabel('Accuracy')
plt.ylim(0, 100)  # Set the y-axis limits to match accuracy values (0 to 100)
plt.show()

