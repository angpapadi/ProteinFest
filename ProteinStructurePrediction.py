import numpy
import os
import random
import subprocess
import sys
import difflib
import Protein
import cPickle as pickle
import matplotlib.pyplot as plt
import keras
import tensorflow
from keras.utils import np_utils

lookup = {'CYS':'C', 'ASP':'D', 'SER': 'S', 'GLN': 'Q', 'LYS': 'K',
          'ILE': 'I', 'PRO': 'P', 'THR': 'T', 'PHE': 'F', 'ASN': 'N',
          'GLY': 'G', 'HIS': 'H', 'LEU': 'L', 'ARG': 'R', 'TRP': 'W',
          'ALA': 'A', 'VAL': 'V', 'GLU': 'E', 'TYR': 'Y', 'MET': 'M'}

random.seed(0)


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

def onehot():
    """Transforms the vocabulary of 20 Aminoacids to one hot vectors"""
    vocab = numpy.array(range(20))
    encoded = np_utils.to_categorical(vocab)
    return encoded

def prepare_data_for_training(objectspath):
    proteins = os.listdir(objectspath)
    numfiles = len(proteins)
    X = []
    Y = []
    random.shuffle(proteins)
    for pfile in proteins:
        # unpickle the protein object
        protein = pickle.load(open(objectspath + pfile, 'rb'))
        X.append(protein.sequence)
        Y.append(protein.ss)

    splitindex = int(0.75 * numfiles)

    Xtrain = X[:splitindex]
    Ytrain = Y[:splitindex]
    Xtest = X[splitindex:]
    Ytest = Y[splitindex:]

    return Xtrain,Ytrain,Xtest,Ytest

def SSclassifier():
    pass

fastalist = 'cullpdb_pc60_res1.8_R0.25_d170805_chains11385.fasta'
sspath = 'STRIDEfiles/'
objectspath = 'ProteinObjs/'
pdbpath = 'PDBfiles/'
cutoff = 9

#init_wrapper(fastalist,sspath,objectspath,pdbpath,cutoff)
[Xtrain, Ytrain, Xtest, Ytest] = prepare_data_for_training(objectspath)
print len(Xtrain),len(Ytrain)
print len(Xtest), len(Ytest)