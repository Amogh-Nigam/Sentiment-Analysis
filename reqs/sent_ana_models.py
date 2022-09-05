#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import string
import re

# digits, len == 0 remove 
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


import contractions
import pkg_resources
from symspellpy import SymSpell
# Remove accented characters
# imports
import unicodedata

import nltk
# from nltk.tokenize import word_tokenize
from nltk.tokenize import TweetTokenizer
Tokeniser = TweetTokenizer()
from nltk.corpus import stopwords
# from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
from nltk.corpus import wordnet
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')


# In[3]:


df = pd.read_csv('./clean_tweet.csv')
df


# In[83]:


df.info()


# In[74]:


df[df['post_clean'].isnull()]
# 80


# In[45]:


df[df['not_post_clean'].isnull()]
# 82


# In[49]:


df[df['post_clean']==""]
# 0


# In[50]:


df[df['not_post_clean']==""]
# 0


# In[90]:


df[df.post_clean.duplicated(keep=False) == True]
# 8469
# md = df.loc[df['post_clean']=='award wapsi gang never return prize money modi']
# md


# In[3]:


df.drop(["clean_text"],axis=1, inplace=True)
df.drop(df[df['post_clean'].isna()].index, inplace=True)
df.drop(df[df['not_post_clean'].isna()].index, inplace=True)
df.drop_duplicates(subset=['category', 'post_clean', 'not_post_clean'], keep='first', inplace=True)
df.reset_index(inplace=True, drop=True)


# In[117]:


df.describe(include='all')


# In[104]:


df.sample(10)


# In[4]:


total = df.category.value_counts()
percentage=round(df.category.value_counts(normalize=True)*100,2)
pd.concat([total,percentage],axis=1,keys=["Total","Percentage"])


# In[127]:


plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["figure.facecolor"] = 'white'

plt.pie(percentage, labels=['Positive','Neutral','Negative'], autopct='%1.1f%%', colors=['lightgreen', 'lightblue', 'red'])
plt.axis('equal')  # Equal aspect ratio ensures the pie chart is circular.
plt.title('Distribution of Data')

plt.show()


# In[5]:


df['post_clean_token']=df['post_clean'].apply(lambda x: len(str(x).split()))
df.head()


# In[6]:


df['not_post_clean_token']=df['not_post_clean'].apply(lambda x: len(str(x).split()))
df.head()


# In[1]:


df.to_csv('cleaner_tweet.csv', header=True, index=False)


# In[4]:


df = pd.read_csv('./cleaner_tweet.csv')


# <h1>post_clean</h1>
# "not" stopword is contained 
# <h1>not_post_clean</h1>
# "not" stopword is NOT contained 

# In[17]:


df.sample(10)


# In[147]:


post_clean_lt = list(df['post_clean'])


# In[5]:


df_negative = df[df["category"]==-1.0].copy()
df_positive = df[df["category"]==1.0].copy()
df_neutral = df[df["category"]==0.0].copy()


# In[8]:


# Wordcloud for neutral tweets
plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["figure.facecolor"] = 'white'
plt.figure(figsize = (20,20))
wc = WordCloud(max_words = 3000 , width = 1600 , height = 800,
               collocations=False).generate(" ".join(df_neutral['post_clean']))
plt.imshow(wc)


# In[9]:


plt.figure(figsize = (20,20))
wc = WordCloud(max_words = 3000 , width = 1600 , height = 800,
               collocations=False).generate(" ".join(df_neutral['not_post_clean']))
plt.imshow(wc)


# In[10]:


# Wordcloud for positive tweets
plt.figure(figsize = (20,20))
wc = WordCloud(max_words = 3000 , width = 1600 , height = 800,
               collocations=False).generate(" ".join(df_positive['post_clean']))
plt.imshow(wc)


# In[11]:


plt.figure(figsize = (20,20))
wc = WordCloud(max_words = 3000 , width = 1600 , height = 800,
               collocations=False).generate(" ".join(df_positive['not_post_clean']))
plt.imshow(wc)


# In[12]:


# Wordcloud for Negative tweets
plt.figure(figsize = (20,20))
wc = WordCloud(max_words = 3000 , width = 1600 , height = 800,
               collocations=False).generate(" ".join(df_negative['post_clean']))
plt.imshow(wc)


# In[13]:


plt.figure(figsize = (20,20))
wc = WordCloud(max_words = 3000 , width = 1600 , height = 800,
               collocations=False).generate(" ".join(df_negative['not_post_clean']))
plt.imshow(wc)


# In[ ]:


# spell check and translation left 


# In[ ]:


# model analysis 
# 1. Naive Bayes
# 2. Support Vector Machines (SVM)
# 3. Random Forests
# 4. XG Boost
# 5. Logistic Regression
# 6. Ensemble O/P


# In[ ]:


# There are 3 types of Naïve Bayes algorithm. The 3 types are listed below:-

# 1. Gaussian Naïve Bayes
# 2. Multinomial Naïve Bayes
# 3. Bernoulli Naïve Bayes


# In[1]:


from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV

from sklearn.model_selection import train_test_split
from scipy.stats import uniform, randint
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, KFold
from sklearn.pipeline import Pipeline

from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score


# max_df is used for removing terms that appear too frequently, also known as "corpus-specific stop words". For example:
# 
# max_df = 0.50 means "ignore terms that appear in more than 50% of the documents".
# max_df = 25 means "ignore terms that appear in more than 25 documents".
# The default max_df is 1.0, which means "ignore terms that appear in more than 100% of the documents". 
# Thus, the default setting does not ignore any terms.
# 
# min_df is used for removing terms that appear too infrequently. For example:
# 
# min_df = 0.01 means "ignore terms that appear in less than 1% of the documents".
# min_df = 5 means "ignore terms that appear in less than 5 documents".
# The default min_df is 1, which means "ignore terms that appear in less than 1 document". 
# Thus, the default setting does not ignore any terms.

# In[4]:


def tf_idf_vec(X_train, X_valid, X_test):
    vectorizer= TfidfVectorizer(sublinear_tf=True, ngram_range=(1,3),)
    tf_x_train = vectorizer.fit_transform(X_train)
    tf_x_valid = vectorizer.transform(X_valid)
    tf_x_test = vectorizer.transform(X_test)
    return tf_x_train, tf_x_valid, tf_x_test


# In[5]:


# Let's say we want to split the data in 70:15:15 for train:valid:test dataset
def split_df(df, split_size, col):
    df2 = df[[col, 'category']].copy()
    X = df2[col]
    y = df2['category']
    # reduce overfitting using validation
    # In the first step we will split the data in training and remaining dataset
    X_train, X_rem, y_train, y_rem = train_test_split(X,y, train_size=split_size, random_state=0)
    # Now since we want the valid and test size to be equal (15% each of overall data). 
    # we have to define valid_size=0.5 (that is 50% of remaining data)
    test_size = 0.5
    X_valid, X_test, y_valid, y_test = train_test_split(X_rem,y_rem, test_size=0.5, random_state=0)
    return X_train, y_train, X_valid, y_valid, X_test, y_test


# In[6]:


X_train, y_train, X_valid, y_valid, X_test, y_test = split_df(df,0.7, 'post_clean')


# In[7]:


X_train_not, y_train_not, X_valid_not, y_valid_not, X_test_not, y_test_not = split_df(df,0.7, 'not_post_clean')


# In[8]:


tf_x_train, tf_x_valid, tf_x_test = tf_idf_vec(X_train, X_valid, X_test)


# In[9]:


tf_x_train_not, tf_x_valid_not, tf_x_test_not = tf_idf_vec(X_train_not, X_valid_not, X_test_not)


# In[14]:


model = MultinomialNB()
model.fit(tf_x_train_not, y_train_not)
y_pred = model.predict(tf_x_valid_not)
cm = confusion_matrix(y_valid_not, y_pred)
print(cm)
print(" ")
print('classification_report: ', classification_report(y_valid, y_pred))


# In[15]:


accuracies = cross_val_score(estimator = model, X = tf_x_train_not, y = y_train_not, cv = 10)   #K-Fold Validation
print('')
print("K-Fold Validation Mean Accuracy: {:.2f} %".format(accuracies.mean()*100))
print('')
print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))


# In[18]:


model = MultinomialNB()
model.fit(tf_x_train, y_train)
y_pred = model.predict(tf_x_valid)
cm = confusion_matrix(y_valid, y_pred)
print(cm)
print(" ")
print('classification_report: ', classification_report(y_valid, y_pred))
accuracies = cross_val_score(estimator = model, X = tf_x_train, y = y_train, cv = 10)   #K-Fold Validation
print('')
print("K-Fold Validation Mean Accuracy: {:.2f} %".format(accuracies.mean()*100))
print('')
print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))


# In[19]:


model = LinearSVC(random_state=0)
model.fit(tf_x_train_not, y_train_not)
y_pred = model.predict(tf_x_valid_not)
cm = confusion_matrix(y_valid_not, y_pred)
print(cm)
print(" ")
print('classification_report: ', classification_report(y_valid_not, y_pred))
accuracies = cross_val_score(estimator = model, X = tf_x_train_not, y = y_train_not, cv = 10)   #K-Fold Validation
print('')
print("K-Fold Validation Mean Accuracy: {:.2f} %".format(accuracies.mean()*100))
print('')
print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))


# In[20]:


model = LinearSVC(random_state=0)
model.fit(tf_x_train, y_train)
y_pred = model.predict(tf_x_valid)
cm = confusion_matrix(y_valid, y_pred)
print(cm)
print(" ")
print('classification_report: ', classification_report(y_valid, y_pred))
accuracies = cross_val_score(estimator = model, X = tf_x_train, y = y_train, cv = 10)   #K-Fold Validation
print('')
print("K-Fold Validation Mean Accuracy: {:.2f} %".format(accuracies.mean()*100))
print('')
print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))


# In[23]:


model = LogisticRegression(max_iter=1000,solver='saga', C = 1.0, random_state=0)
model.fit(tf_x_train_not, y_train_not)
y_pred = model.predict(tf_x_valid_not)
cm = confusion_matrix(y_valid_not, y_pred)
print(cm)
print(" ")
print('classification_report: ', classification_report(y_valid_not, y_pred))
accuracies = cross_val_score(estimator = model, X = tf_x_train_not, y = y_train_not, cv = 10)   #K-Fold Validation
print('')
print("K-Fold Validation Mean Accuracy: {:.2f} %".format(accuracies.mean()*100))
print('')
print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))


# In[24]:


model = LogisticRegression(max_iter=1000,solver='saga',C=1.0, random_state=0)
model.fit(tf_x_train, y_train)
y_pred = model.predict(tf_x_valid)
cm = confusion_matrix(y_valid, y_pred)
print(cm)
print(" ")
print('classification_report: ', classification_report(y_valid, y_pred))
accuracies = cross_val_score(estimator = model, X = tf_x_train, y = y_train, cv = 10)   #K-Fold Validation
print('')
print("K-Fold Validation Mean Accuracy: {:.2f} %".format(accuracies.mean()*100))
print('')
print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))


# In[11]:


model = RandomForestClassifier(random_state=0)
model.fit(tf_x_train_not, y_train_not)
y_pred = model.predict(tf_x_valid_not)
cm = confusion_matrix(y_valid_not, y_pred)
print(cm)
print(" ")
print('classification_report: ', classification_report(y_valid_not, y_pred))
# accuracies = cross_val_score(estimator = model, X = tf_x_train_not, y = y_train_not, cv = 4)   #K-Fold Validation
# print('')
# print("K-Fold Validation Mean Accuracy: {:.2f} %".format(accuracies.mean()*100))
# print('')
# print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))


# In[ ]:


model = RandomForestClassifier(random_state=0)
model.fit(tf_x_train, y_train)
y_pred = model.predict(tf_x_valid)
cm = confusion_matrix(y_valid, y_pred)
print(cm)
print(" ")
print('classification_report: ', classification_report(y_valid, y_pred))
accuracies = cross_val_score(estimator = model, X = tf_x_train, y = y_train, cv = 10)   #K-Fold Validation
print('')
print("K-Fold Validation Mean Accuracy: {:.2f} %".format(accuracies.mean()*100))
print('')
print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))


# In[13]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y_train_not = le.fit_transform(y_train_not)


# In[14]:


model = XGBClassifier(objective='multi:softprob', eval_metric='auc', random_state=0)
model.fit(tf_x_train_not, y_train_not)
y_pred = model.predict(tf_x_valid_not)
cm = confusion_matrix(y_valid_not, y_pred)
print(cm)
print(" ")
print('classification_report: ', classification_report(y_valid_not, y_pred))
# accuracies = cross_val_score(estimator = model, X = tf_x_train_not, y = y_train_not, cv = 10)   #K-Fold Validation
# print('')
# print("K-Fold Validation Mean Accuracy: {:.2f} %".format(accuracies.mean()*100))
# print('')
# print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))


# In[16]:


y_train = le.fit_transform(y_train)


# In[19]:


model = XGBClassifier(n_estimators = 80,max_depth=4, learning_rate=0.2, random_state=0)
model.fit(tf_x_train, y_train)
y_pred = model.predict(tf_x_valid)
cm = confusion_matrix(y_valid, y_pred)
print(cm)
print(" ")
print('classification_report: ', classification_report(y_valid, y_pred))
# accuracies = cross_val_score(estimator = model, X = tf_x_train, y = y_train, cv = 10)   #K-Fold Validation
# print('')
# print("K-Fold Validation Mean Accuracy: {:.2f} %".format(accuracies.mean()*100))
# print('')
# print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))


# In[ ]:





# In[ ]:





# In[ ]:





# In[21]:


def train_cv(model, X_train, y_train, params, n_splits=5, scoring='f1_weighted'):
    kf = KFold(n_splits=n_splits, random_state=0, shuffle=True)

    cv = RandomizedSearchCV(model,
                        params,
                        cv=kf,
                        scoring=scoring,
                        return_train_score=True,
                        n_jobs=-1,
                        verbose=2,
                        random_state=1
                        )
    cv.fit(X_train, y_train)

    print('Best params', cv.best_params_)
    return cv


# In[24]:


svm_parameters = {
    "C":[0.1,1,10],
    "kernel":['linear', 'rbf', 'sigmoid'],
    "gamma":['scale', 'auto']
}


# In[ ]:


mnb_parameters = {'nb__alpha': [0.001, 0.01, 0.1, 1, 10, 100]}


# In[17]:


ls_parameters = {
    'penalty': ['l2', 'l1', 'elasticnet'],
    'C': uniform(scale=10),
    'solver': ['newton-cg', 'lbfgs', 'liblinear', 'saga'],
    'l1_ratio': uniform(scale=10)
    }


# In[ ]:


# chanllenges -> 
# data cleaning
# spell check
# lang translation


# In[ ]:





# In[ ]:




