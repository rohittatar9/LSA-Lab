
# coding: utf-8

# In[1]:

from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords


# In[2]:

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD


# In[3]:

nltk.download('stopwords')


# In[4]:

from sklearn.datasets import fetch_20newsgroups
categories = ['rec.motorcycles']
dataset = fetch_20newsgroups(subset='all',shuffle=True, random_state=42, categories=categories)
corpusdata = dataset.data


# In[5]:

stopset = set(stopwords.words('english'))
stopset.update(['lt','p','/p','br','amp','quot','field','font','normal','span','0px','rgb','style','51', 
                'spacing','text','helvetica','size','family', 'space', 'arial', 'height', 'indent', 'letter'
                'line','none','sans','serif','transform','line','variant','weight','times', 'new','strong', 'video', 'title'
                'white','word','letter', 'roman','0pt','16','color','12','14','21', 'neue', 'apple', 'class', 'nntp', '00 00'  ])


# In[6]:

corpusdata[0]


# In[7]:

vectorizer = TfidfVectorizer(stop_words=stopset,
                                 use_idf=True, ngram_range=(1, 3))
X = vectorizer.fit_transform(corpusdata)


# In[8]:

X[0]


# In[9]:

print X[0]


# In[10]:

X.shape


# In[11]:

lsa = TruncatedSVD(n_components=27, n_iter=100)
lsa.fit(X)


# In[12]:

lsa.components_[0]


# In[13]:

terms = vectorizer.get_feature_names()
for i, comp in enumerate(lsa.components_): 
    termsInComp = zip (terms,comp)
    sortedTerms =  sorted(termsInComp, key=lambda x: x[1], reverse=True) [:10]
    print "Concept %d:" % i
    for term in sortedTerms:
        print term[0]
    print " "


# In[ ]:




# In[ ]:



