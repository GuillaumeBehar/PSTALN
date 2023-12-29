#!/usr/bin/env python3

class UDTagSet():

    def __init__(self):
        self.dico = {
        "ADJ"   : 0, 
        "ADP"   : 1,
        "ADV"   : 2,
        "AUX"   : 3,
        "CCONJ" : 4,
        "DET"   : 5,
        "NOUN"  : 6,
        "NUM"   : 7,
        "PRON"  : 8,
        "PROPN" : 9,
        "PUNCT" : 10,
        "SCONJ" : 11,
        "SYM"   : 12,
        "VERB"  : 13,
        "INTJ"  : 14,
        "X"     : 15,
        "PART"  : 16,
        "_"     : 17,
        "PAD": 18
        }
        self.array = ["ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET", "NOUN", "NUM", "PRON", "PROPN", "PUNCT", "SCONJ", "SYM", "VERB", "INTJ", "X","PART", "_", "PAD"]

    def __getitem__(self, tag):
        return self.dico.get(tag, None)

    def code2tag(self, code):
        if code > (len(self.dico) - 1) or code < 0:
            return None
        else :
            return self.array[code]

    def __len__(self):
        return len(self.dico)        

    def tags(self):
        return self.array;
