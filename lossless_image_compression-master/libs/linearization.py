
import numpy as np
import math

from libs import hilbert_curve as hc
from libs import bwt as bwt

#Images are transposed in data loader to have shape cxwxh instead of wxhxc

def burrows_wheeler_transform(lin_plain_text,block_size=1024):
    #import ipdb;ipdb.set_trace() 
    split_size=block_size
    encoded_text=lin_plain_text.copy()
    indexes=[]
    length=len(lin_plain_text)
    splits=length//split_size+ math.ceil((length%split_size)/split_size)
    start=0 
    for i in range(splits):
        end=min(start+split_size,length-1) 
        split=encoded_text[start:end]
        bwt_ref, idx=bwt.bwt(split)
        encoded_split = [ split[x] for x in bwt_ref ]
        encoded_text[start:end]=encoded_split
        #ibwt_ref = bwt.ibwt(encoded_split, idx)
        #decoded = [encoded_split[x] for x in ibwt_ref]
        indexes=indexes+[idx]
        start+=split_size
    return encoded_text,indexes,split_size



#--------------------Linearization---------------------# 
def linearize_column_major(img) :
    img_lin=[]  
    c,w,h=img.shape 
    for i in range(c):
        for k in range(h):
            for j in range(w):
                img_lin.append(img[i][j][k])
    #print('img.shape: '+str(img.shape)+' pixels: '+str(c*w*h)+' len(img_spiral): '+str(len(img_lin)))
    return img_lin

def linearize_row_major(img) :
    img_lin=[]  
    c,w,h=img.shape
    for i in range(c):
        for j in range(w):
            for k in range(h):
                img_lin.append(img[i][j][k])
    #print('img.shape: '+str(img.shape)+' pixels: '+str(c*w*h)+' len(img_spiral): '+str(len(img_lin)))
    return img_lin

def linearize_snakelike_row_major(img) :
    img_lin=[]  
    c,w,h=img.shape
    for i in range(c):
        direction=1 #1 is rightward and -1 is leftward
        for j in range(w):
            if direction ==1:
                for k in range(h):
                    img_lin.append(img[i][j][k])
            elif direction ==-1:
                for k in reversed(range(h)):
                    img_lin.append(img[i][j][k])
            direction =direction*-1 
    #print('img.shape: '+str(img.shape)+' pixels: '+str(c*w*h)+' len(img_spiral): '+str(len(img_lin)))
    return img_lin


def linearize_spiral_scan(img):
    #import ipdb; ipdb.set_trace()
    img_lin=[]
    c,w,h=img.shape
    for i in range(c):
        direction=0 # 0, 1, 2, 3 
        wl=0
        wu=w
        hl=0
        hu=h
        #wi=wl, hi=hl 
        while wu>=wl or hu >=hl:
            if direction ==0:
                for wi in range(wl,wu,1):
                    img_lin.append(img[i][wi][hl])
                hl+=1
            elif direction ==1:
                for hi in range(hl,hu,1):
                    img_lin.append(img[i][wu-1][hi])
                wu-=1 
            elif direction ==2:
                for wi in reversed(range(wl,wu,1)):
                    img_lin.append(img[i][wi][hu-1])
                hu-=1
            elif direction ==3:
                for hi in reversed(range(hl,hu,1)):
                    img_lin.append(img[i][wl][hi])
                #import ipdb;ipdb.set_trace() 
                wl+=1
            direction=(direction+1)%4
    #print('img.shape: '+str(img.shape)+' pixels: '+str(c*w*h)+' len(img_spiral): '+str(len(img_lin)))
    return img_lin
       

def linearize_hilbert_curve_back(img) :
    c,w,h=img.shape
    img_lin=[]    
    whmax=max(w,h)
    n=int(2**np.ceil(np.log2(max(w,h)))) #next 2th power of max dimension if not power of 2
    #print('img.shape: '+str(img.shape)+' n: '+str(n))
    for i in range(c):
        ch=img[i]
        c_lin=np.zeros(n*n)
        c_lin.fill(-1)#Initialize with -1 and then filter after transform
        #import ipdb;ipdb.set_trace()
        for j in range(w):
            for k in range(h):
                d=hc.xy2d(n,j,k)
                c_lin[d]=ch[j][k]
        img_lin+=list(c_lin[c_lin>=0])        
    #print('img.shape: '+str(img.shape)+' pixels: '+str(c*w*h)+' n: '+str(n)+' len(img_lin): '+str(len(img_lin)))
    return img_lin


def linearize_hilbert_curve(img,hilbert) :
    c,w,h=img.shape
    img_lin=[]    
    whmax=max(w,h)
    for i in range(c):
        ch=img[i]
        c_lin=np.zeros(n*n)
        c_lin.fill(-1)#Initialize with -1 and then filter after transform
        for j in range(w):
            for k in range(h):
                d=hilbert.xy2d(n,j,k)
                c_lin[d]=ch[j][k]
        img_lin+=list(c_lin[c_lin>=0])        
    #print('img.shape: '+str(img.shape)+' pixels: '+str(c*w*h)+' n: '+str(n)+' len(img_lin): '+str(len(img_lin)))
    return img_lin


if __name__=='__main__':
   
    a=np.random.randint(0,80,(2,4,5))
    print(a)
    lin=linearize_spiral_scan(a)
    print(lin)
    print('Done...')


