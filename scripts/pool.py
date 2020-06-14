from multiprocessing import Pool, Lock
from pyaspeller import YandexSpeller
import pyaspeller
import os
import pymorphy2
import io

d = './data/'
files = os.listdir(d)
numthreads = 1
try:
    os.mkdir('./train')
    os.mkdir('./sub')
except:
    n = None  
for i in range(len(files)):
    files[i] = d + files[i]
subf = open('./sample.csv/sample.csv')
sublines = subf.readlines()
sub_qd = {}
sublines.pop(0)
for line in sublines:
    line = line.split(',')
    if line[1][:-1] not in sub_qd.keys():
        sub_qd[line[1][:-1]] = []
    sub_qd[line[1][:-1]].append(line[0])
subf.close()
trf = open('./train.marks.tsv/train.marks.tsv')
tlines = trf.readlines()
tr_qd = {}
for line in tlines:
    line = line.split('\t')
    if line[1] not in tr_qd.keys():
        tr_qd[line[1]] = []
    tr_qd[line[1]].append(line[0])
trf.close()

lock = Lock()

def norm(file):
    global tr_qd
    global sub_qd
    global morphdict
    global d
    idd = file[len(d):-4]
    morph = pymorphy2.MorphAnalyzer()
    ifile = open(file)
    text = " ".join(line[:-1] for line in ifile.readlines())
    ifile.close()
    if idd not in sub_qd.keys() and idd not in tr_qd.keys():
        return
    speller = YandexSpeller(max_requests=555)
    changes = {} 
    newwords = {}
    lock.acquire()
    spelled = speller.spell(text)
    lock.release()
    for change in spelled:
        try:
            changes[change['word']] = change['s'][0] 
        except:
            continue
    for word, suggestion in changes.items():
        try:
            text = text.replace(word, suggestion)
        except:
            continue
    for word in text.split(' '):
        if word not in morphdict.keys():
            newwords[word] = morph.parse(word)[0].normal_form
            try:
                text = text.replace(word, newwords[word])
            except:
                continue
        else:
            try:
                text = text.replace(word, morphdict[word])
            except:
                continue
    if idd in tr_qd.keys():
        for q in tr_qd[idd]:
            try:
                os.mkdir('./train/{}'.format(q))
            except:
                n = None
            ofile = open('./train/{}/{}'.format(q,file),'w')
            ofile.write(text)
            ofile.close()
    if idd in sub_qd.keys():
        for q in sub_qd[idd]:
            try:
                os.mkdir('./sub/{}'.format(q))
            except:
                n = None
            ofile = open('./sub/{}/{}'.format(q,file),'w')
            ofile.write(text)
            ofile.close()
    return newwords


morphdict = {}
while len(files) > 0:
    lfiles = []
    for i in range(numthreads):
        try:
            lfiles.append(files.pop(0))
        except:
            n = None
    p = Pool(len(lfiles))
    ret = p.map(norm,lfiles)
    for dct in ret:
        if dct is not None:
            for key in dct.keys():
                    if key not in morphdict.keys():
                        morphdict[key] = dct[key]
