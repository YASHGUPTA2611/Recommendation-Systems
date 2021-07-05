#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


credits = pd.read_csv(r'C:\Users\The ChainSmokers\Downloads\tmdb_5000_credits.csv')


# In[3]:


credits.head()


# In[4]:


credits.isnull().sum()


# In[5]:


movies_df = pd.read_csv(r'C:\Users\The ChainSmokers\Downloads\tmdb_5000_movies.csv')


# In[6]:


movies_df.head()


# In[7]:


movies_df.isnull().sum()


# In[8]:


credit_column_renamed = credits.rename(index=str,columns={'movie_id':'id'})


# In[9]:


movies_df_merge = movies_df.merge(credit_column_renamed, on='id')


# In[10]:


movies_df_merge.head()


# In[11]:


movies_df_merge.columns


# In[12]:


movies_cleaned_df = movies_df_merge.drop(columns=['homepage','title_x','status', 'production_companies'])


# In[13]:


movies_cleaned_df.info()


# ## Content Based Recommendation System
# We will make an content based recommendation system using summary column. So, if our user gives us a movie title, our goal is to recommend similar movies of that same plot.

# In[14]:


movies_cleaned_df['overview'][0]


# ## Using Natural Language Processing

# In[15]:


from sklearn.feature_extraction.text import TfidfVectorizer


# In[16]:


from sklearn.feature_extraction.text import TfidfVectorizer

tfv = TfidfVectorizer(min_df=3,  max_features=None, 
            strip_accents='unicode', analyzer='word',token_pattern=r'\w{1,}',
            ngram_range=(1, 3),
            stop_words = 'english')

# Filling NaNs with empty string
movies_cleaned_df['overview'] = movies_cleaned_df['overview'].fillna('')


# In[17]:


tfv_matrix = tfv.fit_transform(movies_cleaned_df['overview'])


# In[18]:


tfv_matrix.shape


# In[19]:


from sklearn.metrics.pairwise import sigmoid_kernel

# Compute the sigmoid kernel
sig = sigmoid_kernel(tfv_matrix, tfv_matrix)


# In[26]:


sig.shape


# In[22]:


# Reverse mapping of indices and movie titles
indices = pd.Series(movies_cleaned_df.index, index=movies_cleaned_df['original_title']).drop_duplicates()


# In[23]:


indices


# In[24]:


list(enumerate(sig[indices['Newlyweds']]))


# In[27]:


def give_rec(title, sig=sig):
    # Get the index corresponding to original_title
    idx = indices[title] 

    # Get the pairwsie similarity scores 
    sig_scores = list(enumerate(sig[idx]))

    # Sort the movies 
    sig_scores = sorted(sig_scores, key=lambda x: x[1], reverse=True)

    # Scores of the 10 most similar movies
    sig_scores = sig_scores[1:11]

    # Movie indices
    movie_indices = [i[0] for i in sig_scores]

    # Top 10 most similar movies
    return movies_cleaned_df['original_title'].iloc[movie_indices]


# In[28]:


give_rec('Avatar')


# In[ ]:




