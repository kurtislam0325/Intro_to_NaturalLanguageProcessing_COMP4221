import string 
import random 

import nltk 
import time
import sys


#nltk.download('punkt') 
#nltk.download('stopwords') 
#nltk.download('reuters') 
from nltk.corpus import reuters 
from nltk import FreqDist 
from nltk import ngrams

print("This is an N-Gram Language Model.")

file = open("./text_news.txt", "r")
sents = []
while True:
    content=file.readline()
    if not content:
        break
    sents.append(content)
   # print(content)
file.close()

# input the reuters sentences 
#sents  = ["I like banana and basketball.", "I like banana and apple."] #reuters.sents() 
  
# write the removal characters such as : Stopwords and punctuation 
stop_words = set() #set(stopwords.words('english')) 
string.punctuation = string.punctuation +'"'+'"'+'-'+'''+'''+'—' 
string.punctuation 
removal_list = list(stop_words) + list(string.punctuation)+ ['lt','rt'] 
removal_list 
  
# generate unigrams bigrams trigrams 
unigram=[] 
bigram=[] 
trigram=[] 
tokenized_text=[] 

print("Text Read Finished")


for sentence in sents: 
  #sentence = list(map(lambda x:x.lower(),sentence)) 
  sentence = list(sentence.split())
 # print(sentence)
  for word in sentence: 
        if word== '.': 
            sentence.remove(word)  
        else: 
            unigram.append(word) 
    
  tokenized_text.append(sentence) 
  bigram.extend(list(ngrams(sentence, 2,pad_left=True, pad_right=True))) 
  trigram.extend(list(ngrams(sentence, 3, pad_left=True, pad_right=True))) 
  
# remove the n-grams with removable words 
def remove_stopwords(x):      
    y = [] 
    for pair in x: 
        count = 0
        for word in pair: 
            if word in removal_list: 
                count = count or 0
            else: 
                count = count or 1
        if (count==1): 
            y.append(pair) 
    return (y) 
#unigram = remove_stopwords(unigram) 
#bigram = remove_stopwords(bigram) 
#trigram = remove_stopwords(trigram) 
  
# generate frequency of n-grams  
freq_bi = FreqDist(bigram) 
freq_tri = FreqDist(trigram) 

#print(freq_tri)
  
d = {} #DefaultDict(Counter()) 

for a, b, c in freq_tri: 
  #  print(a, b, c)
    if(a != None and b!= None and c!= None): 
  #    print(a, b, c)
      if (a, b) not in d:
        d[a, b] = {c: freq_tri[a, b, c]}
      elif c in d[a, b]:
        d[a, b][c] += freq_tri[a, b, c] 
      else:
        d[a, b][c] = freq_tri[a, b, c]
        
#print(d)
# Next word prediction       
s='' 
def pick_word(counter): 
    "Chooses a random element."
    return random.choice(list(counter)) 

prefix = ("I", "like")
print(" ".join(prefix)) 
s = " ".join(prefix) 
for i in range(20): 
    suffix = pick_word(d[prefix]) 
    s=s+' '+suffix 
    print(s) 
    prefix = prefix[1], suffix 

s='' 
def pick_word(counter): 
    "Chooses a random element."
    return random.choice(list(counter)) 

context = "It was just a normal summer day when Allan Houston stopped his workout "
prefix = "in", "midchurn"

s = context + " ".join(prefix) 
print(s)

for i in range(30): 
    suffix = pick_word(d[prefix]) 
    
    s=s+' '+suffix 
    print(s)
    
    prefix = prefix[1], suffix 
    
