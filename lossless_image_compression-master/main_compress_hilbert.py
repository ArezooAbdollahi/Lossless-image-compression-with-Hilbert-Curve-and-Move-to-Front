
import numpy as np
import math
import csv
from collections import OrderedDict
import time
from libs.datasets import snu  
from libs import hilbert_curve as hc
from libs import mtf_coding as mtf
from libs.huffman import *


def record_csv(filepath,row):
    with open(filepath,'a') as f:
        writer=csv.writer(f)
        writer.writerow(row)
    return

def calculate_entropy(s):
    #Ref: https://rosettacode.org/wiki/Entropy
    s=list(s)
    hist,lns=np.bincount(s),float(len(s))
    hist=hist[np.nonzero(hist)]
    e=-sum( count/lns*math.log(count/lns,2) for count in hist )
    return e

def linearize_hilbert_curve(img,hilbert) :
    c,w,h=img.shape
    img_lin=[]    
    whmax=max(w,h)
    n=int(2**np.ceil(np.log2(max(w,h)))) #next 2th power of max dimension if not power of 2
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

def compression_pipeline_hilbertcurve_mtf_huffman(data,hilbert):
    encoded=linearize_hilbert_curve(data,hilbert) 
    e1=calculate_entropy(encoded)
    symble_table=list(range(256)) 
    encoded= mtf.MTF_Encode(encoded,symble_table)
    e2=calculate_entropy(encoded)
    hm=HuffmanCoding()
    encoded=hm.compress(encoded)
    er=float(e2)/float(e1)
    compression_output={
            'er':er,
            'compressed_data':encoded             			
            }
    return compression_output

def test_compression(args):
    print('Starting experiment...',args['experiment']) 
    dataset=snu.SNU_DataLoader()
    data_iterator=dataset.iterator()
    hilbert=hc.HilbertCurve(8192)
    header=[]
    header+=['Category','File','Shape','Bytes','Result']
    csv_name_er=args['experiment']+'_entropy_ratio' +'.csv'
    csv_name_cr=args['experiment']+'_compression_ratio' +'.csv'
    csv_name_time=args['experiment']+'_time'+'.csv'  
    record_csv(csv_name_er,header)
    record_csv(csv_name_cr, header )
    record_csv(csv_name_time, header )
    print('starting test ...') 
    for idx,data in enumerate(data_iterator):
        cat,file_name,img=data
        '''if cat not in 'classic': 
        continue'''
        print('idx: '+str(idx)+' cat: '+cat+' file_name: '+file_name+' img.shape: '+str(img.shape))
        c,w,h=img.shape
        result_er=[]        
        result_er.append(cat)
        result_er.append(file_name)
        result_er.append( str(img.shape).replace(',','x'))
        result_er.append(img.shape[0]*img.shape[1]*img.shape[2])			
        result_cr=[]        
        result_cr.append(cat)
        result_cr.append(file_name)
        result_cr.append( str(img.shape).replace(',','x'))
        result_cr.append(img.shape[0]*img.shape[1]*img.shape[2])
        result_time=[]
        result_time.append(cat)
        result_time.append(file_name)
        result_time.append( str(img.shape).replace(',','x'))
        result_time.append(img.shape[0]*img.shape[1]*img.shape[2])		
        #Test all linearization functions...
        name='Hilbert-MTF-HuffMan'
        plaintext_len=img.shape[0]*img.shape[1]*img.shape[2]
        t1=  float("%.9f" % time.time())
        compression_output=compression_pipeline_hilbertcurve_mtf_huffman(img,hilbert)
        t2=  float("%.9f" % time.time())
        t=t2-t1 
        compressed_data=compression_output['compressed_data']
        compressed_data_len=len(compressed_data)
        cr=float(compressed_data_len)/float(plaintext_len)
        er=compression_output['er']
        result_er+=['{:0.2f}'.format(er)]
        result_cr+=['{:0.2f}'.format(cr)]
        result_time+=['{:0.1f}'.format(t)]            
        #------------Record in csv------------------#
        record_csv(csv_name_er, result_er )
        record_csv(csv_name_cr, result_cr )
        record_csv(csv_name_time, result_time )
        #-------------------------------------------#    		
    return



if __name__ == '__main__':
    args={
         'experiment':'snu_hilbert'
         }
    test_compression(args)
    print('The End.')

