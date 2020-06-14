import os
from multiprocessing import Pool
import numpy as np
from nltk.tokenize import word_tokenize
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
data = '../data/'
outdata = './pasT/'
def getpscore(qid):
    global dctqs
    global id_q
    global altits
    texts = {}
    idfdict = {}
    for doc in dctqs[qid]:
        try:
            texts[doc] = word_tokenize(altits[doc])
            for w in texts[doc]:
                if w not in idfdict.keys():
                    idfdict[w] = 0
                idfdict[w] += 1
        except:
            continue
    for key in idfdict.keys():
        idfdict[key] = len(texts) / idfdict[key]
    
    query = word_tokenize(id_q[qid])
    qset = set(query)
    with open(outdata+qid+'.txt','w') as fout:
        for key in texts.keys():
            ltext = texts[key]
            pas = 0
            for i in range(len(ltext) - 2):
                scut = set(ltext[i:i+2])
                ints = set.intersection(scut,qset)
                if len(ints) > 0:
                    cidf = 0
                    for el in ints:
                        cidf+=idfdict[el]
                    for el in ints:
                        for k in ints:
                            if el != k:
                                if (ltext.index(el) - ltext.index(k))*(query.index(el) - query.index(k)) > 0:
                                    cidf *= 1.1
                                if abs((ltext.index(el) - ltext.index(k)))<=abs((query.index(el) - query.index(k))):
                                    cidf *= 1.05
                    pas+=cidf*(len(ltext) - i)
            fout.write(str(key) + '\t' + str(pas) + '\n')
    return

p = Pool(2)
results = p.map(getpscore,[q for q in dctqs.keys()])
