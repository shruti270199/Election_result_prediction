#!/usr/bin/env python
# coding: utf-8

# In[328]:


import re
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
import string
import nltk
import warnings 
warnings.filterwarnings("ignore", category=DeprecationWarning)
from nltk.tokenize.treebank import TreebankWordDetokenizer

get_ipython().run_line_magic('matplotlib', 'inline')
from wordcloud import WordCloud,ImageColorGenerator
from PIL import Image
import urllib
import requests
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


# In[329]:


df=pd.read_csv("C:/Users/user/Desktop/m2/old-data/parties/BJP1.csv",encoding="latin1")
df.shape


# In[386]:


df2=pd.read_csv("C:/Users/user/Desktop/m2/old-data/parties/Congress1.csv",encoding="latin1")
df2.shape


# In[331]:


list5=[]
def sentiment_scores(sentence): 
  
    
    sid_obj = SentimentIntensityAnalyzer() 
  
    # polarity_scores method of SentimentIntensityAnalyzer 
    # oject gives a sentiment dictionary. 
    # which contains pos, neg, neu, and compound scores. 
    sentiment_dict = sid_obj.polarity_scores(sentence) 
      
    #print("Overall sentiment dictionary is : ", sentiment_dict) 
    #print("sentence was rated as ", sentiment_dict['neg']*100, "% Negative") 
    #print("sentence was rated as ", sentiment_dict['neu']*100, "% Neutral") 
    #print("sentence was rated as ", sentiment_dict['pos']*100, "% Positive") 
  
    #print("Sentence Overall Rated As", end = " ") 
  
    # decide sentiment as positive, negative and neutral 
    if sentiment_dict['compound'] >= 0.0 : 
        list5.append(1)
  
     
  
    else : 
        list5.append(-1) 
  


# In[332]:


for i in range(0,13452):
    text=df2["full_text"].values[i]
    sentiment_scores(text)


# In[333]:


df2["Sentiment"]=list5


# In[334]:


train_original=df2.copy()


# In[335]:


test=df.copy()


# In[336]:


combine = df2.append(df,ignore_index=True,sort=True)


# In[337]:


def preprocess_tweet(text):

    # Check characters to see if they are in punctuation
    nopunc = [char for char in text if char not in string.punctuation]
    # Join the characters again to form the string.
    nopunc = ''.join(nopunc)
    # convert text to lower-case
    nopunc = nopunc.lower()
    # remove URLs
    nopunc = re.sub('((www\.[^\s]+)|(https?://[^\s]+)|(http?://[^\s]+))','', nopunc)
    nopunc = re.sub(r'http\S+', '', nopunc)
    # remove usernames
    nopunc = re.sub('@[^\s]+', '', nopunc)
    # remove the # in #hashtag
    nopunc = re.sub(r'#([^\s]+)', r'\1', nopunc)
    # remove repeated characters
    nopunc = word_tokenize(nopunc)
    # remove stopwords from final word list
    return [word for word in nopunc if word not in stopwords.words('english')]


# In[338]:


list1=combine["full_text"]
new = [i for i in list1]
list2=[]
for items in new:
    i=str(items)
    text=preprocess_tweet(i)
    list2.append(text)


# In[339]:


combine['Tidy_Tweets']=list2
combine.shape


# In[340]:


list5=[]
for i in range(0,41265):
    
    list5.append(TreebankWordDetokenizer().detokenize(combine['Tidy_Tweets'].values[i]))


# In[341]:


combine['Tidy_Tweets']=list5


# In[342]:


all_words_positive = ' ,'.join(text for text in combine['Tidy_Tweets'][combine['Sentiment']==1])

all_words_negative = ' ,'.join(text for text in combine['Tidy_Tweets'][combine['Sentiment']==-1])


# In[343]:


# combining the image with the dataset
Mask = np.array(Image.open(requests.get('http://clipart-library.com/image_gallery2/Twitter-PNG-Image.png', stream=True).raw))

# We use the ImageColorGenerator library from Wordcloud 
# Here we take the color of the image and impose it over our wordcloud
image_colors = ImageColorGenerator(Mask)

# Now we use the WordCloud function from the wordcloud library 
wc = WordCloud(background_color='black', height=1500, width=4000,mask=Mask).generate(all_words_positive)
wc


# In[344]:


# combining the image with the dataset
Mask = np.array(Image.open(requests.get('http://clipart-library.com/image_gallery2/Twitter-PNG-Image.png', stream=True).raw))

# We use the ImageColorGenerator library from Wordcloud 
# Here we take the color of the image and impose it over our wordcloud
image_colors = ImageColorGenerator(Mask)

# Now we use the WordCloud function from the wordcloud library 
wc1 = WordCloud(background_color='black', height=1500, width=4000,mask=Mask).generate(all_words_negative)


# In[345]:


plt.figure(figsize=(10,20))

# Here we recolor the words from the dataset to the image's color
# recolor just recolors the default colors to the image's blue color
# interpolation is used to smooth the image generated 
plt.imshow(wc.recolor(color_func=image_colors),interpolation="gaussian")

plt.axis('off')
plt.show()


# In[346]:


plt.figure(figsize=(10,20))

# Here we recolor the words from the dataset to the image's color
# recolor just recolors the default colors to the image's blue color
# interpolation is used to smooth the image generated 
plt.imshow(wc1.recolor(color_func=image_colors),interpolation="gaussian")

plt.axis('off')
plt.show()


# In[347]:


from sklearn.feature_extraction.text import CountVectorizer

bow_vectorizer = CountVectorizer(max_df=0.90, min_df=2, max_features=1000, stop_words='english')

# bag-of-words feature matrix
bow = bow_vectorizer.fit_transform(combine['Tidy_Tweets'])

df_bow = pd.DataFrame(bow.todense())

#df_bow


# In[348]:


from sklearn.feature_extraction.text import TfidfVectorizer

tfidf=TfidfVectorizer(max_df=0.90, min_df=2,max_features=1000,stop_words='english')

tfidf_matrix=tfidf.fit_transform(combine['Tidy_Tweets'])

df_tfidf = pd.DataFrame(tfidf_matrix.todense())


# In[349]:


train_bow = bow[:13452]

#train_bow.todense()


# In[350]:


train_tfidf_matrix = tfidf_matrix[:13452]

#train_tfidf_matrix.todense()


# In[351]:


from sklearn.model_selection import train_test_split


# In[352]:


x_train_bow, x_valid_bow, y_train_bow, y_valid_bow = train_test_split(train_bow,df2['Sentiment'],test_size=0.3,random_state=2)


# In[353]:


x_train_tfidf, x_valid_tfidf, y_train_tfidf, y_valid_tfidf = train_test_split(train_tfidf_matrix,df2['Sentiment'],test_size=0.3,random_state=17)


# In[354]:


from sklearn.linear_model import LogisticRegression
Log_Reg = LogisticRegression(random_state=0,solver='lbfgs')


# In[355]:


Log_Reg.fit(x_train_bow,y_train_bow)
log_bow= Log_Reg.score(x_valid_bow, y_valid_bow)
#print(score)


# In[356]:


Log_Reg.fit(x_train_tfidf,y_train_tfidf)
log_tfidf= Log_Reg.score(x_valid_tfidf, y_valid_tfidf)
#print(score1)


# In[357]:


from xgboost import XGBClassifier


# In[358]:


model_bow = XGBClassifier(random_state=22,learning_rate=0.9)


# In[359]:


model_bow.fit(x_train_bow, y_train_bow)
xgb_bow= model_bow.score(x_valid_bow, y_valid_bow)
#print(score2)


# In[360]:


model_tfidf = XGBClassifier(random_state=29,learning_rate=0.7)


# In[361]:


model_tfidf.fit(x_train_tfidf, y_train_tfidf)
score=model_tfidf.score(x_valid_tfidf, y_valid_tfidf)
print(score)


# In[362]:


from sklearn.naive_bayes import MultinomialNB


# In[363]:


naive_bow= MultinomialNB().fit(x_train_bow, y_train_bow)


# In[364]:


score_naivebow = naive_bow.score(x_valid_bow, y_valid_bow)
print(score_naivebow)
#print(score3)


# In[365]:


naive_tfidf= MultinomialNB().fit(x_train_tfidf, y_train_tfidf)


# In[366]:


naive_tfidf_score=naive_tfidf.score(x_valid_tfidf, y_valid_tfidf)
print(naive_tfidf_score)


# In[367]:


from sklearn.svm import LinearSVC


# In[368]:


svc_model_bow= LinearSVC().fit(x_train_bow, y_train_bow)
score_svcbow = svc_model_bow.score(x_valid_bow, y_valid_bow)
print(score_svcbow)


# In[369]:


svc_tfidf= LinearSVC().fit(x_train_bow, y_train_bow)
svc_tfidf_score=svc_tfidf.score(x_valid_tfidf, y_valid_tfidf)
print(svc_tfidf_score)


# In[370]:


from sklearn.tree import DecisionTreeClassifier
dct = DecisionTreeClassifier(criterion='entropy', random_state=1)


# In[371]:


dct.fit(x_train_bow,y_train_bow)
dct_score_bow = dct.score(x_valid_bow, y_valid_bow)
#print(score5)


# In[372]:


dct.fit(x_train_tfidf,y_train_tfidf)
dct_score_tfidf= dct.score(x_valid_tfidf, y_valid_tfidf)
#print(score6)


# In[373]:


Algo_1 = ['LogisticRegression(Bag-of-Words)','XGBoost(Bag-of-Words)','DecisionTree(Bag-of-Words)','Naive-Bayes(Bag-of-Words)','LinearSVC(Bag-of-Words)']

score_1 = [log_bow,xgb_bow,dct_score_bow,score_naivebow,score_svcbow]

compare_1 = pd.DataFrame({'Model':Algo_1,'Accuracy':score_1},index=[i for i in range(1,6)])

compare_1.T


# In[374]:


plt.figure(figsize=(18,5))

sns.pointplot(x='Model',y='Accuracy',data=compare_1)

plt.title('Bag-of-Words')
plt.xlabel('MODEL')
plt.ylabel('SCORE')

plt.show()


# In[375]:


Algo_2 = ['LogisticRegression(TF-IDF)','XGBoost(TF-IDF)','DecisionTree(TF-IDF)','Naive-Bayes(TF-IDF)','LinearSVC(TF-IDF)']

score_2 = [log_tfidf,score,dct_score_tfidf,naive_tfidf_score,svc_tfidf_score]

compare_2 = pd.DataFrame({'Model':Algo_2,'Accuracy':score_2},index=[i for i in range(1,6)])

compare_2.T


# In[376]:


plt.figure(figsize=(18,5))

sns.pointplot(x='Model',y='Accuracy',data=compare_2)

plt.title('TF-IDF')
plt.xlabel('MODEL')
plt.ylabel('SCORE')

plt.show()


# In[377]:


Algo_best = ['DecisionTree(Bag-of-Words)','DecisionTree(TF-IDF)']

score_best = [dct_score_bow,dct_score_tfidf]

compare_best = pd.DataFrame({'Model':Algo_best,'Accuracy':score_best},index=[i for i in range(1,3)])

compare_best.T


# In[379]:


plt.figure(figsize=(18,5))

sns.pointplot(x='Model',y='Accuracy',data=compare_best)

plt.title('Decision Tree(Bag-of-Words & TF-IDF)')
plt.xlabel('MODEL')
plt.ylabel('SCORE')


# In[394]:


test_tfidf = tfidf_matrix[27813:]
test_pred = dct.predict_proba(df_tfidf)

test_pred_int = test_pred[:,1] >= 0.5
test_pred_int = test_pred_int.astype(np.int)
print(test_pred_int)
print(len(test_pred_int))

combine['Sentiment'] =np.array(test_pred_int)

submission = combine['Sentiment']
combine.to_csv('C:/Users/user/Desktop/m2/old-data/parties/result1.csv', index=False)


# In[57]:


countpbjp=0;
countnbjp=0;
countpcong=0;
countncong=0;
with open("result1.csv", "r") as f:
    csvreader = csv.reader(f)
    for row in csvreader:
        if "Congress" in row[0]:
            if "1" in row[1]:
                countpcong+=1
            else:
                countncong+=1;
        else:
            if "1" in row[1]:
                countpbjp+=1
            else:
                countnbjp+=1;
print(countpbjp)
print(countnbjp)
print(countpcong)
print(countncong)


# In[58]:


objects = ('BJP', 'CONGRESS')
y_pos = np.arange(len(objects))
performance = [countnbjp,countncong]
plt.bar(y_pos, performance, align='center', alpha=1.0)
plt.xticks(y_pos, objects)
plt.ylabel('Negitive tweets')
plt.title('Negitive Sentimental Analysis')

plt.show()


# In[59]:


objects = ('BJP', 'CONGRESS')
y_pos = np.arange(len(objects))
performance = [countpbjp,countpcong]

plt.bar(y_pos, performance, align='center', alpha=1.0)
plt.xticks(y_pos, objects)
plt.ylabel('Positive tweets')
plt.title('Positive Sentimental Analysis')

plt.show()


# In[ ]:




