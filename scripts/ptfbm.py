import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from rank_bm25 import BM25Okapi
import os
from nltk.tokenize import word_tokenize
from multiprocessing import Pool,Lock            
#data 
queries = {}
with open('./norm_q.txt') as fin:
    lines = fin.readlines()
    for line in lines:
        line = line[:-1].split('\t')
        if line[0] == '':
            line.pop(0)
        queries[line[0]] = line[1]

        titles= {}
with open("./norm_tits.txt" ,'r', encoding='utf-8') as fin:
    for line in fin.readlines():
        line=line.split('\t')
        titles[line[0]]= line[1][:-1]

train=[]
train_titles = []
train_queries = []
trq = set()
with open("./train.marks.tsv/train.marks.tsv", 'r') as fin:
    for line in fin.readlines():
        line=line[:-1].split('\t')
        if line[1] in titles.keys():
            train.append([line[0],line[1], line[2]])
            train_titles.append([line[1], titles[line[1]]])
            if line[0] not in trq:
                train_queries.append([line[0], queries[line[0]]])
                trq.add(line[0])

                
test=[]
test_titles = []
test_queries = []
teq = set()
with open("./sample.csv/sample.csv", 'r') as fin:
    fin.readline()
    for line in fin.readlines():
        line=line[:-1].split(',')
        if line[1] in titles.keys():
            test.append([line[0],line[1],-1])
            test_titles.append([line[1], titles[line[1]]])
            if line[0] not in teq:
                test_queries.append([line[0], queries[line[0]]])
                teq.add(line[0])

alq = list(train) + list(test)
dctqs = {}
for q, t, n in alq:
    if q not in dctqs.keys():
        dctqs[q] = []
    dctqs[q].append(t)
alqueries = train_queries + test_queries
id_q = {}
for el in alqueries:
    id_q[el[0]] = el[1]   

altits = {}
for el in train_titles + test_titles:
    altits[el[0]] = el[1]
def cos(sp1,sp2):
    sp1 = sp1.todense()
    sp2 = sp2.todense()
    return np.float64(np.dot(sp1,sp2.T)/np.linalg.norm(sp1)/np.linalg.norm(sp2))
try:
    os.mkdir('./tf')
except:
    n = None


 def ptfidf(q):
     global dctqs
     global id_q
     tf = TfidfVectorizer(analyzer = 'char', ngram_range=(3,7))
     raw = []
     for doc in dctqs[q]:
         try:
             with open('./data/{}.txt'.format(doc)) as fin:
                 raw.append(fin.readline())
         except:
             continue
     raw.append(id_q[q])
     vecs = tf.fit_transform(raw)
     res = []
     for i in range(len(raw) - 1):
         res.append([q,dctqs[q][i], cos(vecs[i],vecs[-1])])
     with open('./tf/{}.txt'.format(q),'w') as fout:
         for el in res:
             fout.write(str(el[0]) + '\t' + str(el[1]) + '\t' + str(el[2]) + '\n')
     return res
 try:
     os.mkdir('./ltf')
 except:
     n = None
 def pltfidf(q):
     global dctqs
     global id_q
     tf = TfidfVectorizer(analyzer = 'char', ngram_range=(1,3))
     raw = []
     for doc in dctqs[q]:
         try:
             with open('./data/{}.txt'.format(doc)) as fin:
                 raw.append(fin.readline())
         except:
             continue
     raw.append(id_q[q])
     vecs = tf.fit_transform(raw)
     res = []
     for i in range(len(raw) - 1):
         res.append([q,dctqs[q][i], cos(vecs[i],vecs[-1])])
     with open('./ltf/{}.txt'.format(q),'w') as fout:
         for el in res:
             fout.write(str(el[0]) + '\t' + str(el[1]) + '\t' + str(el[2]) + '\n')
     return res

 try:
     os.mkdir('./newbm')
 except:
     n = None
       
 def pbm(q):
     global dctqs
     global id_q
     raw = []
     for doc in dctqs[q]:
         try:
             with open('./data/{}.txt'.format(doc)) as fin:
                 raw.append([doc,word_tokenize(fin.readline())])
         except:
             continue
     print(raw, )
     bm = BM25Okapi([doc for idd, doc in raw])
     scores = bm.get_scores(id_q[q])
     with open('./newbm/'+str(q)+'.txt','w') as fout:
         for i, score in enumerate(scores):
             fout.write(str(raw[i][0]) + '\t' + str(score) + '\n')
     return
                            

 p = Pool(16)
 results = p.map(pbm,[q for q,t in alqueries if str(q) + '.txt' not in os.listdir('./newbm')])

