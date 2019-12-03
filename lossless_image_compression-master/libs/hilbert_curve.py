'''
Reference:  https://en.wikipedia.org/wiki/Hilbert_curve . 
Hilber space filling curve on nxn 2d space with n being power of 2. 
(x,y) coordinates (0,0) at bottom left and (n-1,n-1) in top right.
d is 0 at lower left and n^2 -1 on lower right. 
---@RK, 26th October, 2018
'''


import numpy as np
import time

class HilbertCurve():

    def __init__(self,xmax=32):
        self.xmax=xmax
        self.xy2d_table=self.precalculate_table(n=self.xmax)
        return

    def precalculate_table(self,n):
        print(time.ctime())
        print('initializing lookup table for H: Begins')
        #import ipdb;ipdb.set_trace()
        table= np.zeros(shape=(n,n),dtype=int) 
        for x in range(n):
            for y in range(n):
                d=self.xy2d_online(n,x,y)
                table[x][y]=d
        print('initializing lookup table for H: Done')
        print(time.ctime())
        return table  

    def xy2d (self,n,x, y) :
        ''' Uses lookup table'''
        if n> self.xmax:
            xmax=int(2**np.ceil(np.log2(n))) #next 2th power of max dimension if not power of 2
            self.xy2d_table=self.precalculate_table(n=xmax)
            self.xmax=xmax
        d=self.xy2d_table[x][y] 
        return d

    def xy2d_online (self, n,  x, y) :
        '''calculate directly without table '''
        d=0
        s=int(n//2)
        while s>0:
            #import ipdb;ipdb.set_trace()
            rx=int( ( x & s)>0)
            ry= int((y& s)> 0)
            d+=s*s*((3*rx)^ry)
            x,y=self.rot(s,x,y,rx,ry)
            s=s//2   
        return d

    def d2xy(self, n, d):
        '''calculate directly without table '''
        s=1
        t=d
        x=0
        y=0
        while s<n:
            rx=1& (t/2)
            ry=1& (t^rx)
            x,y=self.rot(s, x, y, rx, ry)
            x=x+s*rx
            y=y+s*ry
            t=t/4
            s*=2
        return x, y

    #rotate/flip a quadrant appropriately
    def rot(self, n, x, y,  rx, ry):
        if ry == 0: 
            if rx == 1: 
                x = n-1 - x;
                y = n-1 - y;
            #Swap x and y
            t  = x;
            x = y;
            y = t;
        return x, y    

if __name__=='__main__':
    n=16
    h=HilbertCurve(n) 
    for x in range(8):
        for y in range(8):
            d1=h.xy2d(n,x,y)
            d2=h.xy2d_online(n,x,y)
            print('x=%d, y=%d, d1=%d, d2=%d'%(x,y,d1,d2))
    import ipdb;ipdb.set_trace()
    print('Done')
 
   
