
import numpy as np
import math



'''
def entropy(s):
    #Ref: https://rosettacode.org/wiki/Entropy
    s=list(s)
    p,lns=np.nonzero(np.bincount(s))[0],float(len(s))
    e=-sum( count/lns*math.log(count/lns,2) for count in p )
    return e
'''

def calculate_entropy(s):
    #Ref: https://rosettacode.org/wiki/Entropy
    s=list(s)
    hist,lns=np.bincount(s),float(len(s))
    hist=hist[np.nonzero(hist)]
    e=-sum( count/lns*math.log(count/lns,2) for count in hist )
    return e


if __name__=='__main__':
    print('---testing entropy---')
    s='1223334444'
    print('s: '+str(s)+' entropy(s)[should be 1.8464393446710154 ]: '+str(calculate_entropy(s)))


