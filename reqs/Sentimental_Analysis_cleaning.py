#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import string
import re

# digits, len == 0 remove 
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,confusion_matrix,classification_report
from sklearn.feature_extraction.text import TfidfVectorizer

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


# In[170]:


# Importing the dataset
df = pd.read_csv('./Twitter_Data.csv')
df


# In[171]:


# 162968
print('Rows     :',df.shape[0])
print('Columns  :',df.shape[1])


# In[5]:


df.sample(10)


# In[6]:


df.info()


# In[6]:


df.describe()


# In[7]:


df['category'].value_counts()


# In[8]:


df.isnull().sum()


# In[9]:


df[df['category'].isnull()]


# In[16]:


df[df['clean_text'].isnull()]


# In[10]:


df[df.clean_text.duplicated(keep=False) == True]


# In[13]:


df.loc[df['clean_text'] == " "]


# In[ ]:


# DROP AS VERY LESS
# 4+7+1 = 12


# In[172]:


# make a function def clean
df.drop(df[df['clean_text'].isna()].index, inplace=True)
df.drop(df[df['category'].isna()].index, inplace=True)
df.drop(df[df['clean_text'] == " "].index, inplace = True)
df.reset_index(inplace=True, drop=True)


# In[15]:


df.category.value_counts()


# In[173]:


total = df.category.value_counts()
percentage=round(df.category.value_counts(normalize=True)*100,2)
pd.concat([total,percentage],axis=1,keys=["Total","Percentage"])


# In[ ]:


# Insights

# 1. In this data, we have more than 40% positive tweets
# 2. Negative Tweets are with low numbers and only 50% in count compared to positive tweets
# 3. Neutral Tweets have a good number in total between positive & negative tweet count


# In[11]:


# plt.rcParams.update({"figure.figsize": (10, 6),
#               "figure.facecolor" : "white"
#              })
plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["figure.facecolor"] = 'white'

ax = df.groupby('category').count().plot(kind='bar', title='Distribution of Data',legend=False)
ax.set_xticklabels(['Negative','Neutral','Positive'], rotation=0)
plt.ylabel('Count')
plt.xlabel('Types of Sentiment')

plt.show()


# In[12]:


# fig, ax = plt.subplots()

plt.pie(percentage, labels=['Positive','Neutral','Negative'], autopct='%1.1f%%', colors=['lightgreen', 'lightblue', 'red'])
plt.axis('equal')  # Equal aspect ratio ensures the pie chart is circular.
plt.title('Distribution of Data')

plt.show()


# In[ ]:


# Word count


# In[174]:


df['word_counts']=df['clean_text'].apply(lambda x: len(str(x).split()))
df.head()


# In[175]:


df['chars_count']=df['clean_text'].apply(lambda x: len(x))
df.head()


# In[36]:


df.describe()


# In[24]:


plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["figure.facecolor"] = 'white'
plt.title('Distribution of number of tokens in tweets')
sns.boxplot(x = df['word_counts'])
plt.show()


# In[25]:


plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["figure.facecolor"] = 'white'
plt.title('Distribution of number of tokens in tweets')
sns.boxplot(x = df['chars_count'])
plt.show()


# In[178]:


texts_list, cat_list = list(df['clean_text']), list(df['category'])
df_negative = df[df["category"]==-1.0].copy()
df_positive = df[df["category"]==1.0].copy()
df_neutral = df[df["category"]==0.0].copy()


# In[179]:


texts_len = [len(str(t).split()) for t in texts_list]


# In[180]:


len(texts_list)


# In[183]:


def tokenised_sen(text):
    for i in range (len(text)):
        new = Tokeniser.tokenize(text[i])
        text[i] = " ".join(new)


# In[184]:


texts_tokenised = texts_list.copy()


# In[185]:


tokenised_sen(texts_tokenised)


# In[ ]:


# EMOJI analysis


# In[186]:


# import emot
import emoji
# emot_obj = emot.core.emot() 


# In[187]:


def split_count(textm):
    emoji_list = []
    for i in textm:
        dic = emoji.emoji_list(i)
        if dic:
            emoji_list += [j['emoji'] for j in dic]
    return emoji_list


# In[188]:


cat_emojis = {1.0: [], 0.0: [], -1.0: []}

for i, text in enumerate(texts_tokenised):
    emoji_count = split_count(text)
    if emoji_count:
        cat_emojis[df['category'].iloc[i]].extend(emoji_count)


# In[189]:


plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["figure.facecolor"] = 'white'

for t, emote in cat_emojis.items():
    plt.figure(figsize=(10, 5))
    bar_info = pd.Series(emote).value_counts()[:20]
    print('============'*10,  f'\n\t\t\t\tTop emojis for {t} \n', list(bar_info.index))
    bar_info.index = [emoji.demojize(i, delimiters=("", "")) for i in bar_info.index]
    sns.barplot(x=bar_info.values, y=bar_info.index)
    plt.title(f'{t}')
    plt.show()


# In[ ]:


# 1. decode emoji
# 2. remove num/mentions/url/punc
# 0. lower
# 3. expand contractions
# # 4. translate
# 5. spell check
# 6. remove stopwords
# 7. remove extra spaces
# 8. Lemmatisation


# In[190]:


sym_spell = SymSpell(max_dictionary_edit_distance=3)
dictionary_path = pkg_resources.resource_filename(
    "symspellpy", "frequency_dictionary_en_82_765.txt")
sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)


# In[217]:


# stp_words = stopwords.words('english') + list(STOPWORDS)


# In[220]:


stp_words = stopwords.words('english')


# In[219]:


stopWords_with_not = stp_words
print(stopWords_with_not)
# append "shall"

# try remove not then model analysis 
# with not removing not then model analysis 


# In[82]:


stopWords_with_not.extend(['shall'])
print(stopWords_with_not)


# In[81]:


stopWords_without_not = stp_words
stopWords_without_not.remove('not')
# stopWords_without_not.remove("not")
# stopWords_without_not.remove("cannot")
print(stopWords_without_not)


# In[83]:


stopWords_without_not.extend(['shall'])
print(stopWords_without_not)


# In[222]:


negations_dic = {"isn't": 'is not', "aren't": 'are not', "wasn't": 'was not', "weren't": 'were not', 
                 "needn't": 'need not', "haven't": 'have not', "hasn't": 'has not', "hadn't": 'had not', 
                 "won't": 'will not', "shan't": 'shall not', "wouldn't": 'would not', "don't": 'do not', 
                 "doesn't": 'does not', "didn't": 'did not', "can't": 'can not', "couldn't": 'could not', 
                 "shouldn't": 'should not', "mightn't": 'might not', "mustn't": 'must not', 'isnt': 'is not', 
                 'arent': 'are not', 'wasnt': 'was not', 'werent': 'were not', 'neednt': 'need not', 'havent': 
                 'have not', 'hasnt': 'has not', 'hadnt': 'had not', 'wont': 'will not', 'shant': 'shall not', 
                 'wouldnt': 'would not', 'dont': 'do not', 'doesnt': 'does not', 'didnt': 'did not', 
                 'cant': 'can not', 'couldnt': 'could not', 'shouldnt': 'should not', 'mightnt': 'might not', 
                 'mustnt': 'must not'}

contraction = {'cuz':'because', 'abt':'about'}

def combine_tokens(tokenized): 
    non_tokenized = ' '.join([w for w in tokenized])
    return non_tokenized

def make_tokens(text):
    return Tokeniser.tokenize(text)

def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts"""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}

    return tag_dict.get(tag, wordnet.NOUN)


# In[227]:


# DATA CLEANING
# Remove mentions 
regex_mentions = r"@[^\s]+"
# Remove links -> URL
regex_links = r"((http://)[^ ]*|(https://)[^ ]*|( www\.)[^ ]*)"
# Remove some special characters
regex_special = r"[^A-Z a-z ]+"
# 161335
# Remove Email -> mail -> 0
regex_email = r"([a-zA-Z0-9+_-]+@[a-zA-Z0-9._-]+\.[a-zA-Z0-9_-]+)"
# Remove repeated sequence of letters
sequencePattern   = r"(.)\1\1+"
seqReplacePattern = r"\1"
# remove html
# from bs4 import BeautifulSoup
# df['clean_text']=df['clean_text'].apply(lambda x: BeautifulSoup(x,'lxml').get_text())
def decode_emojis(texts):
#     print('Decoding emojis...')
    return emoji.demojize(texts, language='en', delimiters = (" "," "))

def regex_text(texts):
    texts = re.sub(regex_email, " ", texts.lower())
    texts = re.sub(regex_mentions, " ", texts.lower())
    texts = re.sub(regex_links, " ", texts.lower())
    texts = re.sub(regex_special, " ", texts.lower())
    texts = re.sub(sequencePattern, seqReplacePattern, texts.lower())
    return texts

# function to remove accented characters
def remove_acc_chars(text):
    new_text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    return new_text

def expand_contractions(text):
    for key in contraction:
        value=contraction[key]
        text = text.replace(key,value)
    new_text = contractions.fix(text)
    return new_text

def spellchecker(texts):
    texts = sym_spell.word_segmentation(texts)
    return texts.corrected_string

def rem_stopwords(texts,stop_words):
    tokenized = make_tokens(texts)
    tokenized = [token for token in tokenized if token not in stop_words]
    texts = combine_tokens(tokenized)
    return texts

def remove_double_space(texts):
    pattern = re.compile(' +')
    return re.sub(pattern, ' ', texts)

def lemmatise(text):
    tokenized = make_tokens(text)
#     lemmatizer = WordNetLemmatizer()
    tokenized = [lemmatizer.lemmatize(w, get_wordnet_pos(w)) for w in tokenized]
    tokenized = [token for token in tokenized if token.isalpha() and len(token)>2]
    return tokenized

# def lang_to_eng(texts)


# In[166]:


df2 = df.copy()


# In[ ]:





# In[ ]:


# 1. decode emoji
# 2. remove num/mentions/url/punc
# 0. lower
# 3. expand contractions
# # 4. translate
# # 5. spell check
# 6. remove stopwords
# 7. remove extra spaces
# 8. Lemmatisation


# In[228]:


import time
start_time = time.time()
df2['post_clean'] = df2['clean_text'].apply(lambda x: decode_emojis(x))
df2['post_clean'] = df2['post_clean'].apply(lambda x: x.lower())
df2['post_clean'] = df2['post_clean'].apply(lambda x: regex_text(x))
df2['post_clean'] = df2['post_clean'].apply(lambda x: remove_acc_chars(x))
df2['post_clean'] = df2['post_clean'].apply(lambda x: expand_contractions(x))
df2['post_clean'] = df2['post_clean'].apply(lambda x: remove_double_space(x))
stop_time = time.time()
print("Done")
# 1 min


# In[229]:


print(f'Expansion of all tweets takes ~{round((start_time-stop_time), 3)} seconds')


# In[232]:


start_time = time.time()
# df2['post_clean'] = df2['post_clean'].apply(lambda x: spellchecker(x))
df2['post_clean'] = df2['post_clean'].apply(lambda x: rem_stopwords(x, stopWords_without_not))
df2['not_post_clean'] = df2['post_clean'].apply(lambda x: rem_stopwords(x, stopWords_with_not))

df2['post_clean'] = df2['post_clean'].apply(lambda x: remove_double_space(x))
df2['not_post_clean'] = df2['not_post_clean'].apply(lambda x: remove_double_space(x))

df2['post_clean'] = df2['post_clean'].apply(lambda x: lemmatise(x))
df2['post_clean'] = df2['post_clean'].apply(lambda x: combine_tokens(x))

df2['not_post_clean'] = df2['not_post_clean'].apply(lambda x: lemmatise(x))
df2['not_post_clean'] = df2['not_post_clean'].apply(lambda x: combine_tokens(x))
stop_time = time.time()
print("Done")
# 5 min


# In[233]:


print(f'Lemmatisation of all tweets takes ~{round((start_time-stop_time), 3)} seconds')


# In[234]:


df2.to_csv('clean_tweet.csv', header=True, index=False)


# In[235]:


df = pd.read_csv('./clean_tweet.csv')
df


# In[243]:


df['clean_text'][156571]


# In[241]:


df['post_clean'][162800]


# In[171]:


df['clean_text'][3521]


# In[38]:


cat_list[3521]


# In[179]:


df['clean_text'][353]


# In[ ]:




