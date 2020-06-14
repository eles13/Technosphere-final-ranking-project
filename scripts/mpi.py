from pyaspeller import YandexSpeller
from mpi4py import MPI
import pyaspeller
import os
import pymorphy2
import io
d = './data/'
files = os.listdir(d)

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
if rank == 0:
    try:
        os.mkdir('./train')
        os.mkdir('./sub')
    except:
        n = None    
    fs = []
    for i in range(size):
        fs.append([])
    for i,file in enumerate(files):
        fs[i % size].append(d+file)
    for i in range(size):
        comm.send(fs[i],i,0)
lfiles = []
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
lfiles = comm.recv()
morph = pymorphy2.MorphAnalyzer()
morphdict = {}
for file in lfiles:
    ifile = open(file)
    text = " ".join(line[:-1] for line in ifile.readlines())
    ltext = text.split('\t')
    header = ltext[1]
    text = ltext[2]
    ifile.close()
    if ltext[0] not in sub_qd.keys() and ltext[0] not in tr_qd.keys():
        continue
    speller = YandexSpeller()
    changes = {}
    for change in speller.spell(text):
        try:
            changes[change['word']] = change['s'][0] 
        except:
            continue
    for change in speller.spell(header):
        try:
            changes[change['word']] = change['s'][0] 
        except:
            continue
    recbuf = {}
    newwords = {}
    for sugg in changes.values():
        if sugg not in morphdict.keys():
            morphdict[sugg] = morph.parse(sugg)[0].normal_form
            newwords[sugg] = morphdict[sugg]
    for word, suggestion in changes.items():
        try:
            text = text.replace(word, morphdict[suggestion])
        except:
            continue
    for word, suggestion in changes.items():
        try:
            header = header.replace(word, morphdict[suggestion])
        except:
            continue
    for i in range(size):
        if i != rank:
            comm.isend(newwords,i,rank)
    for i in range(size):
        if i != rank:
            recbuf = comm.recv(None,i,i)
            for key in recbuf.keys():
                if key not in morphdict.keys():
                    morphdict[key] = recbuf[key]
    if ltext[0] in tr_qd.keys():
        for q in tr_qd[ltext[0]]:
            try:
                os.mkdir('./train/{}'.format(q))
            except:
                n = None
            ofile = open('./train/{}/{}.txt'.format(q,ltext[0]),'w')
            ofile.write(ltext[0] + '\t' + header+ '\t' + text)
            ofile.close()
    if ltext[0] in sub_qd.keys():
        for q in sub_qd[ltext[0]]:
            try:
                os.mkdir('./sub/{}'.format(q))
            except:
                n = None
            ofile = open('./sub/{}/{}.txt'.format(q,ltext[0]),'w')
            ofile.write(ltext[0] + '\t' + header+ '\t' + text)
            ofile.close()
    comm.Barrier()
MPI.Finalize()
