#!/usr/bin/env python
# coding: utf-8

# In[6]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[7]:


comments = pd.read_csv(r'Downloads\UScomments.csv',error_bad_lines=False)


# In[8]:


comments.head()


# In[8]:


comments.isnull() #checking for missing values


# In[9]:


comments.isnull().sum() #checking for missing values nnder each feature


# In[10]:


comments.dropna(inplace=True)


# In[11]:


comments.isnull().sum() 


# In[12]:


get_ipython().system('pip install textblob')


# In[13]:


from textblob import TextBlob


# In[15]:


comments.head(6)


# In[17]:


TextBlob("Logan Paul it's yo big day ‼️‼️‼️").sentiment


# In[18]:


TextBlob("Logan Paul it's yo big day ‼️‼️‼️").sentiment.polarity


# In[24]:


sample_df =comments[0:1000]


# In[25]:


sample_df.shape


# In[27]:


polarity = []

for comment in comments['comment_text']:
    try:
        polarity.append(TextBlob(comment).sentiment.polarity)
    except:
        polarity.append(0)


# In[28]:


comments['polarity'] = polarity


# In[30]:


comments.tail(7)


# In[20]:


len(comments)


# In[21]:


comments.shape


# In[32]:


filter1 = comments['polarity'] == 1


# In[37]:


comments_positive = comments[filter1]


# In[34]:


filter2 = comments['polarity'] == -1


# In[35]:


comments_negative = comments[filter2]


# In[38]:


comments_positive.head(5)


# In[39]:


comments_negative.head(6)


# In[40]:


get_ipython().system('pip install wordcloud')


# In[41]:


from wordcloud import WordCloud , STOPWORDS


# In[43]:


set(STOPWORDS)


# In[51]:


comments["comment_text"]


# In[52]:


type(comments["comment_text"])


# In[59]:


wordcloud = WordCloud(stopwords=set(STOPWORDS)).generate(total_comments_positive)


# In[64]:


plt.imshow(wordcloud)
plt.axis('off')


# #converting to string

# In[54]:


' '.join(comments_positive["comment_text"])


# In[56]:


total_comments_positive = ' '.join(comments_positive["comment_text"])


# In[57]:


type(total_comments_positive)


# In[65]:


total_comments_negative = ' '.join(comments_negative["comment_text"])


# In[66]:


type(total_comments_negative)


# In[67]:


wordcloud = WordCloud(stopwords=set(STOPWORDS)).generate(total_comments_negative)


# In[68]:


plt.imshow(wordcloud)
plt.axis('off')


# In[ ]:




