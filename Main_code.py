
import os
import numpy as np
import pandas as pd
pd.options.display.float_format = '{:,.2f}'.format
import re



from nltk.tokenize import RegexpTokenizer, word_tokenize, wordpunct_tokenize
from nltk.corpus import wordnet
import spacy 
  
# Load English tokenizer, tagger,  
# parser, NER and word vectors 
nlp = spacy.load("en_core_web_sm") 


np.set_printoptions(precision=2, suppress=True)


# ## Reading Datasets

# In[17]:


train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
sample = pd.read_csv('sample_submission.csv')


# In[18]:


tokenizer = re.compile('([\s.,;:()]+)')


# In[19]:


train.head()


# In[20]:


test.head()


# ## This will be first approach

# In[21]:


train['answer_text_broken'] = train['answer_text'].str.lower()
train['answer_text_broken'] = train['answer_text_broken'].apply(lambda x : re.split(tokenizer, x))


# In[22]:


''.join(train['answer_text'][1])


# In[23]:


def syn_ant(word):
    ant = list()
    syn = list()

    for synset in wordnet.synsets(word):
        for lemma in synset.lemmas():
            syn.append(lemma.name())    #add the synonyms
            if lemma.antonyms():    #When antonyms are available, add them into the list
                ant.append(lemma.antonyms()[0].name())
    ant = list(set(ant))
    syn = list(set(syn))
    
    return syn[:4], ant


# ## Commencing 3 stage prediction TESTING

# In[53]:


test['answer_text_broken'] = test['answer_text'].str.lower()
test['answer_text_broken'] = test['answer_text_broken'].apply(lambda x : re.split(tokenizer, x))


# In[54]:


centre = test['answer_text_broken'].values
done = [0 for _ in range(len(centre))]
dist = [[] for _ in range(len(centre))]



#Going to execute 7 stage approach for all the 2 distractors, as one of the distractors will be None of the above
for i in range(len(centre)):
    
    #Stage 1 Changing numbers in the options!
    for j in range(len(centre[i])):
        try:
            int(centre[i][j])
            centre[i][j] = str(int(centre[i][j]) + 1)
            dist[i].append(''.join(centre[i]))
            centre[i][j] = str(int(centre[i][j]) + 1)
            dist[i].append(''.join(centre[i]))
            dist[i].append('No of the above')
            done[i] += 3
            break
        except:
            pass
          

        
    #Stage 2 Changing all the proposition in the sentence by antonyms  
    if done[i] == 0:
        st2 = nlp(test['answer_text'].values[i])
        for j in range(len(st2)):
            if st2[j].pos_ == 'PROPN':
                if len(syn_ant(str(st2[j]))[1])>2:
                    c1, c2 = str(st2), str(st2)
                    an1, an2 = syn_ant(str(st2[j]))[1][0], syn_ant(str(st2[j]))[1][1] 
                    c1 = c1.replace(str(st2[j]), an1)
                    c2 = c2.replace(str(st2[j]), an2)
                    dist[i].append(c1)
                    dist[i].append(c2)
                    dist[i].append('None of the above')
                    done[i]+=3
                    break
                    
                elif len(syn_ant(str(st2[j]))[1]) == 1:
                    c1, c2 = str(st2), str(st2)
                    an1, an2 = syn_ant(str(st2[j]))[1][0], syn_ant(str(st2[j]))[0][0] 
                    c1 = c1.replace(str(st2[j]), an1)
                    c2 = c2.replace(str(st2[j]), an2)
                    dist[i].append(c1)
                    dist[i].append(c2)
                    dist[i].append('None of the above')
                    done[i]+=3
                    break
                    
                elif len(syn_ant(str(st2[j]))[0]) >= 2:
                    c1, c2 = str(st2), str(st2)
                    an1, an2 = syn_ant(str(st2[j]))[0][0], syn_ant(str(st2[j]))[0][1] 
                    c1 = c1.replace(str(st2[j]), an1)
                    c2 = c2.replace(str(st2[j]), an2)
                    dist[i].append(c1)
                    dist[i].append(c2)
                    dist[i].append('None of the above')
                    done[i]+=3
                    break
                    
    #Stage 3 Changing all the adjectives        
    if done[i] == 0:
        st2 = nlp(train['answer_text'].values[i])
        for j in range(len(st2)):
            if st2[j].pos_ == 'ADJ':
                if len(syn_ant(str(st2[j]))[1])>2:
                    c1, c2 = str(st2), str(st2)
                    an1, an2 = syn_ant(str(st2[j]))[1][0], syn_ant(str(st2[j]))[1][1] 
                    c1 = c1.replace(str(st2[j]), an1)
                    c2 = c2.replace(str(st2[j]), an2)
                    dist[i].append(c1)
                    dist[i].append(c2)
                    dist[i].append('None of the above')
                    done[i]+=3
                    break
                    
                elif len(syn_ant(str(st2[j]))[1]) == 1:
                    c1, c2 = str(st2), str(st2)
                    an1, an2 = syn_ant(str(st2[j]))[1][0], syn_ant(str(st2[j]))[0][0] 
                    c1 = c1.replace(str(st2[j]), an1)
                    c2 = c2.replace(str(st2[j]), an2)
                    dist[i].append(c1)
                    dist[i].append(c2)
                    dist[i].append('None of the above')
                    done[i]+=3
                    break
                    
                elif len(syn_ant(str(st2[j]))[0]) >= 2:
                    c1, c2 = str(st2), str(st2)
                    an1, an2 = syn_ant(str(st2[j]))[0][0], syn_ant(str(st2[j]))[0][1] 
                    c1 = c1.replace(str(st2[j]), an1)
                    c2 = c2.replace(str(st2[j]), an2)
                    dist[i].append(c1)
                    dist[i].append(c2)
                    dist[i].append('None of the above')
                    done[i]+=3
                    break
    #Stage 4 Changing the conjunctions!               
    if done[i] == 0:
        st2 = nlp(test['answer_text'].values[i])
        for j in range(len(st2)):
            if st2[j].pos_ == 'CCONJ' or st2[j].pos_ == 'CONJ':
                c1, c2 = str(st2), str(st2)
                chan = False
                if c1 == 'AND':
                    c1 = 'OR'
                    c2 = 'NOR'
                    chan = True
                elif c1 == 'OR':
                    c1 = 'AND'
                    c2 = 'NOR'
                    chan = True
                if c2 == 'AND':
                    c2 = 'OR'
                    c1 = 'NOR'
                    chan = True
                elif c2 == 'OR':
                    c2 = 'AND'
                    c1 = 'NOR'
                    chan = True
                if chan==False:
                    c1 ,c2= 'AND', 'OR'
                c1 = c1.replace(str(st2[j]), an1)
                c2 = c2.replace(str(st2[j]), an2)
                dist[i].append(c1)
                dist[i].append(c2)
                dist[i].append('None of the above')
                done[i]+=3
                break
                
    #Stage 5 Changing the adpositions!               
    if done[i] == 0:
        st2 = nlp(test['answer_text'].values[i])
        for j in range(len(st2)):
            if st2[j].pos_ == 'ADP':
                if len(syn_ant(str(st2[j]))[1])>2:
                    c1, c2 = str(st2), str(st2)
                    an1, an2 = syn_ant(str(st2[j]))[1][0], syn_ant(str(st2[j]))[1][1] 
                    c1 = c1.replace(str(st2[j]), an1)
                    c2 = c2.replace(str(st2[j]), an2)
                    dist[i].append(c1)
                    dist[i].append(c2)
                    dist[i].append('None of the above')
                    done[i]+=3
                    break
                    
                elif len(syn_ant(str(st2[j]))[1]) == 1:
                    c1, c2 = str(st2), str(st2)
                    an1, an2 = syn_ant(str(st2[j]))[1][0], syn_ant(str(st2[j]))[0][0] 
                    c1 = c1.replace(str(st2[j]), an1)
                    c2 = c2.replace(str(st2[j]), an2)
                    dist[i].append(c1)
                    dist[i].append(c2)
                    dist[i].append('None of the above')
                    done[i]+=3
                    break
                    
                elif len(syn_ant(str(st2[j]))[0]) >= 2:
                    c1, c2 = str(st2), str(st2)
                    an1, an2 = syn_ant(str(st2[j]))[0][0], syn_ant(str(st2[j]))[0][1] 
                    c1 = c1.replace(str(st2[j]), an1)
                    c2 = c2.replace(str(st2[j]), an2)
                    dist[i].append(c1)
                    dist[i].append(c2)
                    dist[i].append('None of the above')
                    done[i]+=3
                    break
                    
    # Stage 6 Changing Nouns                
    if done[i] == 0:
        st2 = nlp(test['answer_text'].values[i])
        for j in range(len(st2)):
            if st2[j].pos_ == 'NOUN':
                if len(syn_ant(str(st2[j]))[1])>2:
                    c1, c2 = str(st2), str(st2)
                    an1, an2 = syn_ant(str(st2[j]))[1][0], syn_ant(str(st2[j]))[1][1] 
                    c1 = c1.replace(str(st2[j]), an1)
                    c2 = c2.replace(str(st2[j]), an2)
                    dist[i].append(c1)
                    dist[i].append(c2)
                    dist[i].append('None of the above')
                    done[i]+=3
                    break
                    
                elif len(syn_ant(str(st2[j]))[1]) == 1:
                    c1, c2 = str(st2), str(st2)
                    an1, an2 = syn_ant(str(st2[j]))[1][0], syn_ant(str(st2[j]))[0][0] 
                    c1 = c1.replace(str(st2[j]), an1)
                    c2 = c2.replace(str(st2[j]), an2)
                    dist[i].append(c1)
                    dist[i].append(c2)
                    dist[i].append('None of the above')
                    done[i]+=3
                    break
                    
                elif len(syn_ant(str(st2[j]))[0]) >= 2:
                    c1, c2 = str(st2), str(st2)
                    an1, an2 = syn_ant(str(st2[j]))[0][0], syn_ant(str(st2[j]))[0][1] 
                    c1 = c1.replace(str(st2[j]), an1)
                    c2 = c2.replace(str(st2[j]), an2)
                    dist[i].append(c1)
                    dist[i].append(c2)
                    dist[i].append('None of the above')
                    done[i]+=3
                    break
    
    # Stage 7, Changing Adverbs
    if done[i] == 0:
        st2 = nlp(test['answer_text'].values[i])
        for j in range(len(st2)):
            if st2[j].pos_ == 'ADV':
                if len(syn_ant(str(st2[j]))[1])>2:
                    c1, c2 = str(st2), str(st2)
                    an1, an2 = syn_ant(str(st2[j]))[1][0], syn_ant(str(st2[j]))[1][1] 
                    c1 = c1.replace(str(st2[j]), an1)
                    c2 = c2.replace(str(st2[j]), an2)
                    dist[i].append(c1)
                    dist[i].append(c2)
                    dist[i].append('None of the above')
                    done[i]+=3
                    break
                    
                elif len(syn_ant(str(st2[j]))[1]) == 1:
                    c1, c2 = str(st2), str(st2)
                    an1, an2 = syn_ant(str(st2[j]))[1][0], syn_ant(str(st2[j]))[0][0] 
                    c1 = c1.replace(str(st2[j]), an1)
                    c2 = c2.replace(str(st2[j]), an2)
                    dist[i].append(c1)
                    dist[i].append(c2)
                    dist[i].append('None of the above')
                    done[i]+=3
                    break
                    
                elif len(syn_ant(str(st2[j]))[0]) >= 2:
                    c1, c2 = str(st2), str(st2)
                    an1, an2 = syn_ant(str(st2[j]))[0][0], syn_ant(str(st2[j]))[0][1] 
                    c1 = c1.replace(str(st2[j]), an1)
                    c2 = c2.replace(str(st2[j]), an2)
                    dist[i].append(c1)
                    dist[i].append(c2)
                    dist[i].append('None of the above')
                    done[i]+=3
                    break


# In[56]:


np.unique(done, return_counts=True)

test.drop(['answer_text_broken'], axis=1, inplace=True)

test['distractor'] = dist

test['distractors'] = test['distractor'].apply(lambda x: "’,’".join(x))

test['distractors'] = test['distractors'].apply(lambda x: "'" + x + "'")

test.drop(['distractor'], axis=1, inplace=True)

test.columns = ['question', 'answer_text', 'distractor']

test.to_csv('submission_shivam13juna@gmail.csv', index=False)


# In[ ]:




