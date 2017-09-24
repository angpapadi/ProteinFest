class Protein:

    def __init__(self, pdb, chain, seq, nres, ss = '', cm = None):
        self.chainid = chain
        self.sequence = seq
        self.pdbid = pdb
        self.nres = nres
        self.cm = cm
        self.ncontacts = 0  # not reliable yet
        self.ss = ss


