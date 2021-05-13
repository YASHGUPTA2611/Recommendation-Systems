#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importing all important libraries which will be in use
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ### Reading all the datasets using Pandas library

# In[2]:


ratings = pd.read_csv(r'C:\Users\The ChainSmokers\Desktop\book recommendation\BX-Book-Ratings.csv', sep=';',error_bad_lines=False, encoding='latin-1')


# In[3]:


ratings.head()


# In[4]:


ratings.columns


# In[5]:


users = pd.read_csv(r'C:\Users\The ChainSmokers\Desktop\book recommendation\BX-Users.csv', sep=';',error_bad_lines=False, encoding='latin-1')


# In[6]:


users.head()


# In[7]:


books= pd.read_csv(r'C:\Users\The ChainSmokers\Desktop\book recommendation\BX-Books.csv', sep=';',error_bad_lines=False, encoding='latin-1')


# In[8]:


books.head()


# ### Countplot of the Number of Ratings

# In[9]:


ratings['Book-Rating'].value_counts(sort=False).plot(kind='bar')


# ### Histogram of the people's age who have given ratings

# In[10]:


users.Age.hist(bins=[0, 10, 20, 30, 40, 50, 100])


# ### Recommendation based on rating counts

# In[11]:


rating_counts = pd.DataFrame(ratings.groupby('ISBN')['Book-Rating'].count())
rating_counts.sort_values(ascending=False, by='Book-Rating')


# On the basis of the Rating counts we cannot recommend the books because we don't know that the highest book-rating count has the highest rating.  

# In[12]:


# Getting ISBN number of top 5 highest book-rating counts
most_rated_books = pd.DataFrame(['0971880107', '0316666343', '0385504209', '0060928336','0312195516'], columns=['ISBN'])


# In[13]:


most_rated_books


# In[14]:


# Getting names of top 5 books havings highest book-rating counts
most_rated_books_summary = pd.merge(most_rated_books, books, on='ISBN')


# In[15]:


most_rated_books_summary


# ### Recommendation based on correlation

# In[16]:


average_rating = pd.DataFrame(ratings.groupby("ISBN")['Book-Rating'].mean())
average_rating['ratingcount'] = pd.DataFrame(ratings.groupby('ISBN')['Book-Rating'].count())
average_rating.sort_values('ratingcount', ascending=False).head()


# On the basis of correlation we cannot recommnend the books because the books having the book-rating count high has the lowest rating. 

# ### To make our recommendation good we will apply some Statistical Significance.

# By using some statistical study, we will take the users which gave more than 200 and books more have 100 ratings

# In[17]:


counts1 = ratings['User-ID'].value_counts()
ratings = ratings[ratings['User-ID'].isin(counts1[counts1 >= 200].index)]
counts = ratings['Book-Rating'].value_counts()
ratings = ratings[ratings['Book-Rating'].isin(counts[counts >= 100].index)]


# ### Making a Pivot table(Rating Matrix)

# In[18]:


ratings_pivot = ratings.pivot(index='User-ID', columns='ISBN').Book-Rating
userid = ratings_pivot.index
ISBN = ratings_pivot.columns


# Now we take any book of ISBN number and finding the correlation with other books.

# In[ ]:


bones_ratings = ratings_pivot['0316666343']
similar_to_bones = ratings_pivot.corrwith(bones_ratings)
corr_bones = pd.DataFrame(similar_to_bones, columns=['pearsonR'])
corr_bones.dropna(inplace=True)
corr_summary = corr_bones.join(average_rating['ratingCount'])
corr_summary[corr_summary['ratingCount']>=300].sort_values('pearsonR', ascending=False).head(10)


# Getting the titles of the books to recommend for the above ISBN number

# In[ ]:


books_corr_to_bones = pd.DataFrame(['0312291639', '0316601950', '0446610038', '0446672211', '0385265700', '0345342968', '0060930535', '0375707972', '0684872153'], index=np.arange(9), columns=['ISBN'])
corr_books = pd.merge(books_corr_to_bones, books, on='ISBN')
corr_books


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




