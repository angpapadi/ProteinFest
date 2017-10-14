import numpy
import os
import random
import subprocess
import sys
import difflib
import Protein
import cPickle as pickle
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Embedding
from keras.layers import wrappers
from keras.layers import Bidirectional
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras import optimizers
from sklearn.utils import shuffle
#import tensorflow
from keras.utils import np_utils
from keras.layers.normalization import BatchNormalization

os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=gpu,floatX=float32"

lookup = {'CYS':'C', 'ASP':'D', 'SER': 'S', 'GLN': 'Q', 'LYS': 'K',
          'ILE': 'I', 'PRO': 'P', 'THR': 'T', 'PHE': 'F', 'ASN': 'N',
          'GLY': 'G', 'HIS': 'H', 'LEU': 'L', 'ARG': 'R', 'TRP': 'W',
          'ALA': 'A', 'VAL': 'V', 'GLU': 'E', 'TYR': 'Y', 'MET': 'M'}

pslookup = {'D':1,'S':2, 'Q':3, 'K': 4, 'I':5,'P':6, 'T':7, 'F':8,
            'N':9, 'G':10,'H':11,'L':12,'R':13,'W':14,'A':15,'V':16, 'E':17,
            'Y':18,'M':19, 'C':20}

sslookup = {'H':[1,0,0,0,0,0,0,0],
            'T':[0,1,0,0,0,0,0,0],
            'E':[0,0,1,0,0,0,0,0],
            'G':[0,0,0,1,0,0,0,0],
            'B':[0,0,0,0,1,0,0,0],
            'I':[0,0,0,0,0,1,0,0],
            'C':[0,0,0,0,0,0,1,0],
            'b':[0,0,0,0,0,0,0,1]}

callback = [
    EarlyStopping(monitor='val_loss', patience=3, verbose=2)
]

VOCABSIZE = len(lookup.keys())+1
EMBEDIM = 100
NEPOCHS = 10
NHIDDEN = 300
LR = 0.05
BATCHSIZE = 32
SEED = 0

random.seed(SEED)

def visualize_cm(contact_map):
    # plots the given contact map
    plt.imshow(contact_map,interpolation= 'none', cmap='gray')
    plt.show()

def extract_coordinates(filename, chainidf):
    f = open('PDBfiles/' + filename[0:4].lower() + '.pdb', 'r')
    alldata = f.read()
    f.close()
    alldata = alldata.split('\n')

    CAcoords = []
    CBcoords = []
    terflag = False
    for line in alldata:
        if line[:4] == 'ATOM':
            if line[21] == chainidf:
                if line[13:15] == 'CA':
                    CAcoords.append(line)
                    terflag = True
                if line[13:15] == 'CB':
                    CBcoords.append(line)
                    terflag = True
        if line[:3] == 'TER':
            if terflag == True:
                break

    return CAcoords, CBcoords

def residue_distance(res1, res2):

    r1 = numpy.array(res1)
    r2 = numpy.array(res2)
    dist = numpy.linalg.norm(r1-r2)
    return dist

def parselist(filename):
    # read fasta list to create pdbid list
    # return a dictionary that maps every protein to the corresponding chain
    f = open(filename, 'r')
    fastalist = f.read()
    f.close()
    fastalist = fastalist.replace('\n', '')
    fastalist = fastalist.split('>')
    fastalist.pop(0)

    pdbid_chain = {}
    pdbids = ''
    for protein in fastalist:
        chain = protein[4]
        id = protein[0:4]
        pdbids = pdbids + ' ' + id
        pdbid_chain[id] = chain

    # store pdbids in a txt file in order to retrieve the respective pdb files
    f = open('PDBids.txt', 'w')
    f.write(pdbids)
    f.close()

    return pdbid_chain

def createProteinObjs(sspath, objectspath):
    ssfiles = os.listdir(sspath)

    for filename in ssfiles:
        fi = open(sspath + filename, 'r')
        sscontent = fi.read()
        sscontent = sscontent.split('\n')
        fi.close()

        ss = ''
        chainid = ''
        for l in range(len(sscontent)):
            line = sscontent[l]
            # detect sequence in the secondary structure file
            if line[0:3] == 'CHN' and len(chainid)==0:
                chainid = line.split()[2]
                step = 3
                seq = sscontent[l+step][10:60]
                step += 4
                while sscontent[l+step][0:3] == 'SEQ':
                    seq = seq + sscontent[l+step][10:60]
                    step += 4
                seq= "".join(seq.split())

            # extract secondary structure
            if line[0:3] != 'ASG':
                continue
            if line[9]!= chainid:
                continue
            ss+=line[24]

        # create protein pickle
        destinationfile = filename[0:4] + '.txt'
        nres = len(seq)
        cm = numpy.full((nres, nres), 999)
        pobj = Protein.Protein(filename[0:4], chainid, seq, nres, ss, cm)
        pickle.dump(pobj, open(objectspath + destinationfile, 'wb'))

def runStride(pdb_chain_dic, destpath):

    for pdbid in pdb_chain_dic.keys():
        pdbfilename = 'PDBfiles/'+ pdbid.lower() +'.pdb'
        readchainarg = '-r'+pdb_chain_dic[pdbid]
        processchainarg = '-c'+pdb_chain_dic[pdbid]
        outputfilename = '-f'+ destpath + pdbid + '.txt'
        subprocess.call(['stride/stride', pdbfilename, readchainarg, processchainarg, outputfilename])

def fillmaps(pdbpath, objectspath, cutoff):
    # create native contact maps from pdb coordinates
    pdbfiles = os.listdir(pdbpath)
    for i in range(len(pdbfiles)):
        # for every protein extract atom coordinates from pdb file
        pfile = pdbfiles[i]
        filename = pfile[0:4].upper() + '.txt'
        # retrieve pickled protein object
        if (filename not in os.listdir(objectspath)):
            continue
        prot = pickle.load(open(objectspath + filename, 'rb'))
        chainidf = prot.chainid
        cm = prot.cm
        sequence = prot.sequence
        nres = prot.nres

        CAcoords, CBcoords = extract_coordinates(filename, chainidf)

        # for every pair of residues compute euclidean distance
        countera = 0
        counterb = 0
        flag = False
        for k in range(nres):
            type = 'B'
            currentres = sequence[k]
            # while the observed residues are not over
            if countera < len(CAcoords):
                currentCA = CAcoords[countera]
                if currentCA[17:20] not in lookup.keys():
                    continue
                # if the expected and observed residues match keep going, else continue to the next residue
                if currentres == lookup[currentCA[17:20]]:
                    if counterb < len(CBcoords):

                        # if the residue is glycine (no CB atom) change distance type to count only CA atoms
                        if currentres != 'G':
                            currentCB = CBcoords[counterb]
                            counterb += 1
                        else:
                            type = 'A'
                        countera += 1
                    else:
                        continue
                else:
                    if flag == False:
                        flag = True
                    continue
            else:
                continue

            # for every otherresidue
            counteraa = 0
            counterbb = 0
            for j in range(nres):
                # if they are the same residue skip to the next
                if k == j:
                    counteraa += 1
                    if currentres != 'G':
                        counterbb += 1
                    continue
                otherres = sequence[j]
                if counteraa < len(CAcoords):
                    otherCA = CAcoords[counteraa]
                    if otherCA[17:20] not in lookup.keys():
                        continue
                    if otherres == lookup[otherCA[17:20]]:
                        if counterbb < len(CBcoords):
                            if otherres != 'G':
                                otherCB = CBcoords[counterbb]
                                counterbb += 1
                            else:
                                type = 'A'
                            counteraa += 1
                        else:
                            continue
                    else:
                        continue
                else:
                    continue

                # calculate distance
                if type == 'AB':
                    currentCOORDS = (float(currentCA[30:38]) + float(currentCB[30:38]),
                                     float(currentCA[38:46]) + float(currentCB[38:46]),
                                     float(currentCA[46:54]) + float(currentCB[46:54]))
                    otherCOORDS = (float(otherCA[30:38]) + float(otherCB[30:38]),
                                   float(otherCA[38:46]) + float(otherCB[38:46]),
                                   float(otherCA[46:54]) + float(otherCB[46:54]))
                elif type == 'A':
                    currentCOORDS = (float(currentCA[30:38]),
                                     float(currentCA[38:46]),
                                     float(currentCA[46:54]))
                    otherCOORDS = (float(otherCA[30:38]),
                                   float(otherCA[38:46]),
                                   float(otherCA[46:54]))
                elif type == 'B':
                    currentCOORDS = (float(currentCB[30:38]),
                                     float(currentCB[38:46]),
                                     float(currentCB[46:54]))
                    otherCOORDS = (float(otherCB[30:38]),
                                   float(otherCB[38:46]),
                                   float(otherCB[46:54]))

                distance = residue_distance(currentCOORDS, otherCOORDS)

                # update contact map with the appropritate distance
                if distance <= cutoff:
                    cm[k][j] = distance

        prot.cm = cm
        prot.ncontacts = numpy.where(cm != 999)[0].size
        pickle.dump(prot, open(objectspath + filename, 'wb'))

        # visualize native cm
        #visualize_cm(cm)

def init_wrapper(fastalist, sspath, objectspath, pdbpath, cutoff):
    pdbid_chain_dict = parselist(fastalist)     # parse fasta list to create pdbid-chain dictionary
    runStride(pdbid_chain_dict, sspath)         # execute bash script to run stride for all pdb files
    createProteinObjs(sspath, objectspath)      # extract ss info & create protein objects
    fillmaps(pdbpath, objectspath, cutoff)      # generate native contact maps from pdb coordinates
    prepare_data_for_training(objectspath)      # prepare dataset for training

def onehot():
    """Transforms the vocabulary of 20 Aminoacids to one hot vectors"""
    vocab = numpy.array(range(20))
    encoded = np_utils.to_categorical(vocab)
    dict  = {}
    i = 0
    for key in lookup.values():
        dict[key] = encoded[i]
        i= i+1
    return dict

def prepare_data_for_training(objectspath):
    proteins = os.listdir(objectspath)
    X = []
    Y = []

    for pfile in proteins:
        # unpickle the protein object
        protein = pickle.load(open(objectspath + pfile, 'rb'))

        # encode every aminoacid by an integer
        psec = protein.sequence
        ssec = protein.ss
        encodedpsec = []
        encodedssec = []

        for idx in range(len(psec)):
            encodedpsec.append(pslookup.get(psec[idx],0))
            encodedssec.append(sslookup[ssec[idx]])

        X.append(encodedpsec)
        Y.append(encodedssec)

    # assert equal length by zeropadding
    maxsseqlen = len(max(X, key=len))
    for i in range(len(X)):
        l = len(X[i])
        Y[i].extend([[0, 0, 0, 0, 0, 0, 0, 0]] * (maxsseqlen - l))
        X[i].extend([0] * (maxsseqlen - l))

    X = numpy.array(X)
    Y = numpy.array(Y)

    dataset = [X,Y]
    pickle.dump(dataset, open('dataset', 'wb'))

def prepareCB513(cb513path):
    # prepares the CB513 dataset for validation of the SSclassifier model
    cb513files = os.listdir(cb513path)
    data = []
    labels = []

    for filename in cb513files:
        fi = open(cb513path + filename, 'r')
        content = fi.read()
        content = content.split('\n')
        fi.close()

        residuesequence = content[0]
        stridesequence = content[3]
        residuesequence = residuesequence.replace(',','')
        stridesequence = stridesequence.replace(',','')
        residuesequence = residuesequence[4:]
        stridesequence = stridesequence[7:]


        if (len(residuesequence) == len(stridesequence))== False:
            print len(residuesequence), len(stridesequence)
            continue

        # encode residues and secondary structure information as integers
        sequence = []
        ssinfo = []
        for index in range(len(residuesequence)):
            residue = residuesequence[index]
            ss = stridesequence[index]
            sequence.append(pslookup.get(residue,0))
            ssinfo.append(sslookup.get(ss,sslookup['C']))

        data.append(sequence)
        labels.append(ssinfo)

    # assert equal length by zeropadding
    maxseqlen = len(max(data, key=len))
    for i in range(len(data)):
        l = len(data[i])
        labels[i].extend([[0, 0, 0, 0, 0, 0, 0, 0]] * (maxseqlen - l))
        data[i].extend([0] * (maxseqlen - l))

    #data = numpy.array(data)
    #labels = numpy.array(labels)

    return [data,labels]

def SSclassifier():
    dataset = pickle.load(open('dataset', 'rb'))
    X = dataset[0][:]
    Y = dataset[1][:]

    # shuffle data and labels in a consistent way
    X, Y = shuffle(X, Y, random_state = SEED)

    # split into training and test set
    splitindex = int(0.8 * len(X))
    Xtrain = X[:splitindex]
    Ytrain = Y[:splitindex]
    Xtest = X[splitindex:]
    Ytest = Y[splitindex:]

    embeddinglayer = Embedding(VOCABSIZE, EMBEDIM, mask_zero = True)
    lstmlayer = LSTM(NHIDDEN, return_sequences=True)
    densehidden = wrappers.TimeDistributed(Dense(150))
    outputlayer = wrappers.TimeDistributed(Dense(8, activation='softmax'))

    model = Sequential()
    model.add(embeddinglayer)
    model.add(Bidirectional(lstmlayer))
    model.add(BatchNormalization())
    model.add(densehidden)
    model.add(outputlayer)
    #model.add(Dense(100))
    #model.add(Dense(8, activation='softmax'))

    sgd = optimizers.SGD(lr=LR, momentum=0.0, decay=0.0, nesterov=False)
    adam = optimizers.Adam(lr=LR, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    opt = adam

    model.compile(optimizer= opt, loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    history = model.fit(Xtrain, Ytrain, nb_epoch=NEPOCHS, batch_size=BATCHSIZE, verbose=2, validation_data=(Xtest,Ytest))

    # plot model metrics
    #plt.plot(history.history['categorical_accuracy'])
    #plt.plot(history.history['val_categorical_accuracy'])
    #plt.plot(history.history['loss'])
    #plt.plot(history.history['val_loss'])
    #plt.legend(['train acc', 'test acc', 'train loss', 'test loss'], loc='upper left')
    #plt.show()

    # save trained model to a file
    model.save('SSclassifiermasked3.h5')

def ContactPredictor():
    pass

def testonCB513():
    [data,labels]= prepareCB513(cb513path)
    model = load_model('SSclassifiermasked2.h5')
    scores = model.evaluate( data, labels, batch_size=1, verbose=0, sample_weight=None)
    print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

fastalist = 'cullpdb_pc60_res1.8_R0.25_d170805_chains11385.fasta'
sspath = 'STRIDEfiles/'
objectspath = 'ProteinObjs/'
pdbpath = 'PDBfiles/'
cb513path = 'CB513/'
cutoff = 9

#init_wrapper(fastalist,sspath,objectspath,pdbpath,cutoff)

SSclassifier()
#testonCB513()


