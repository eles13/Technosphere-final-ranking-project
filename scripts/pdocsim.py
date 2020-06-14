from gensim.models.doc2vec import TaggedDocument, Doc2Vec
import numpy as np
import os
import operator
from multiprocessing import Pool,Lock
from nltk.tokenize import word_tokenize
from sklearn.metrics.pairwise import cosine_similarity as cosine
queries = {}
with open('./norm_q.txt') as fin:
    lines = fin.readlines()
    for line in lines:
        line = line[:-1].split('\t')
        if line[0] == '':
            line.pop(0)
        queries[line[0]] = line[1]
file =open("./norm_tits.txt" ,'r', encoding='utf-8')
titles= {}
for line in file.readlines():
    l=line.split('\t')
    titles[l[0]]= l[1].strip()
file.close()
file = open("./train.marks.tsv", 'r')
train=[]
train_titles = []
train_queries = []
set_q= []
for l in file.readlines():
    splits=l.split('\t')
    if splits[1] in titles.keys():
        train.append([splits[0],splits[1], splits[2][:-1]])
        train_titles.append([splits[1], titles[splits[1]]])
        if splits[0] not in set_q:
            train_queries.append([splits[0], queries[splits[0]]])
            set_q.append(splits[0])
file.close()
def cos(sp1,sp2):
    return np.float64(np.dot(sp1,sp2.T)/np.linalg.norm(sp1)/np.linalg.norm(sp2))
file = open("./sample.csv", 'r')
file.readline()
test = []
test_titles = []
test_queries = []
set_q = []
for l in file.readlines():
    splits=l.split(',')
    if splits[1][:-1] in titles.keys():
        test.append([splits[0],splits[1][:-1],-1])
        test_titles.append([splits[1], titles[splits[1][:-1]]])
        if splits[0] not in set_q:
            test_queries.append([splits[0], queries[splits[0]]])
            set_q.append(splits[0])
file.close()
alq = list(train) + list(test)
dctqs = {}
for q, t, n in alq:
    if q not in dctqs.keys():
        dctqs[q] = []
    dctqs[q].append(t)

def getfiles(lfiles):
    for file in lfiles:
        try:
            with open('./data/{}.txt'.format(file)) as fin:
                text = fin.readline()
            yield file,text
        except:
            continue

            
try:
    os.mkdir('./docsim')
except:
    n = None
def doc_2_vec_sim(qid):
    max_epochs = 7
    alpha = 0.0025
    model = Doc2Vec(dm=0, vector_size=100, negative=5, hs=0, min_count=2, sample = 0)
    tagged_data = [TaggedDocument(words=word_tokenize(doc), tags=[j]) for j,doc in getfiles(dctqs[qid])]
    model.build_vocab(tagged_data)
    for epoch in range(max_epochs):
        model.train(tagged_data,
        total_examples=model.corpus_count,
        epochs=model.iter)
        # decrease the learning rate
        model.alpha -= 0.0002
        # fix the learning rate, no decay
        model.min_alpha = model.alpha
    with open('./docsim/{}.txt'.format(qid),'w') as fout:
        for outdoc in dctqs[qid]:
            fout.write(str(outdoc) + '\t')
            ldict = {}
            for doc in dctqs[qid]:
                if doc != outdoc:
                    try:
                        ldict[doc] = cos(model.docvecs[doc],model.docvecs[outdoc])
                    except:
                        continue
            sdict = sorted(ldict.items(), key = operator.itemgetter(1), reverse = True)
            for i in range(15):
                try:
                    fout.write(str(sdict[i][0]) + ' ' + str(sdict[i][1]) + '\t')
                except:
                    continue
            fout.write('\n')
    return 
p = Pool(16)
p.map(doc_2_vec_sim, [q for q in dctqs.keys() if str(q) + '.txt' not in os.listdir('./docsim')])
