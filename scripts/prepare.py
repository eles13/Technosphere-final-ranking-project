import os
gf = os.listdir('./data')
def getline(docs):
    for doc in docs:
        with open('./data/'+doc) as fin:
            line = fin.readline()
            while line != '':
                line = line[:-1].split('\t')
                key = line[0]
                line.pop(0)
                line = " ".join(line)
                yield key, line
                line = fin.readline()
prdir = './prdata/'
for key, text in getline(gf):
    with open(prdir+key+'.txt','w') as fout:
        fout.write(text)
