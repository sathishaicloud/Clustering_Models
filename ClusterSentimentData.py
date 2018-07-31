print("k-mean clustering starts...")

import csv

with open("C:/dev/MachineLearning/Clustering/K-means/sentiment labelled sentences/imdb_labelled.txt","r") as text_file:
    lines = text_file.read().split('\n')

lines = [line.split("\t") for line in lines if len(line.split("\t"))==2 and line.split("\t")[1] != '']

#for line in lines:
 #   if(len(line.split("\t"))==2 and line.split("\t")[1] != ''):
  #      print(line) 

train_doc = [line[0] for line in lines]

#print(train_doc)

# get the machine learning alg.

from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_vectorzier = TfidfVectorizer(max_df=0.5,min_df=2,stop_words='english')
train_doc = tfidf_vectorzier.fit_transform(train_doc)

#print(train_doc)

from sklearn.cluster import KMeans

km = KMeans(n_clusters=3,init='k-means++',max_iter=100,n_init=1,verbose=True)
km.fit(train_doc)

#km = KMeans(copy_x=True,n_clusters=3,init='k-means++',max_iter=100,n_init=1,verbose=True)

count=0
for i in range(len(lines)):
    #if (count>3):
     #   break
    if km.labels_[i]==3:
        print(lines[i])
        #count+=1
