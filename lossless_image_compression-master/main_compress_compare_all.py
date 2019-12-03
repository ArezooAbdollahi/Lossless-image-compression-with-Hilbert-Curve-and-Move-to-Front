
import numpy as np
import math
import csv
from collections import OrderedDict
import time
from libs.datasets import snu  
from libs import compression_pipelines as cp

def record_csv(filepath,row):
    with open(filepath,'a') as f:
        writer=csv.writer(f)
        writer.writerow(row)
    return


def is_power2(num):
    'states if a number is a power of two'
    return num != 0 and ((num & (num - 1)) == 0)


def test_compression(args):
    print('Starting experiment...',args['experiment']) 
    dataset=snu.SNU_DataLoader()
    data_iterator=dataset.iterator()
    compression_pipelines=OrderedDict()
    compression_pipelines['column_major_huffman']=cp.column_major_huffman
    compression_pipelines['column_major_mtf_huffman']=cp.column_major_mtf_huffman
    compression_pipelines['column_major_burrows_mtf_huffman']=cp.column_major_burrows_mtf_huffman
    compression_pipelines['row_major_mtf_huffman']=cp.row_major_mtf_huffman
    compression_pipelines['row_major_burrows_mtf_huffman']=cp.row_major_burrows_mtf_huffman
    compression_pipelines['snakelike_row_major_mtf_huffman']=cp.snakelike_row_major_mtf_huffman
    compression_pipelines['snakelike_row_major_burrows_mtf_huffman']=cp.snakelike_row_major_burrows_mtf_huffman
    compression_pipelines['spiral_outer_mtf_huffman']=cp.spiral_outer_mtf_huffman
    compression_pipelines['spiral_outer_burrows_mtf_huffman']=cp.spiral_outer_burrows_mtf_huffman
    compression_pipelines['hilbertcurve_mtf_huffman']=cp.hilbertcurve_mtf_huffman
    compression_pipelines['hilbertcurve_burrows_mtf_huffman']=cp.hilbertcurve_burrows_mtf_huffman
    header=[]
    header+=['Category','File','Shape','Bytes']
    for k in compression_pipelines.keys():
        header+=[k]
    #print(header)           
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
        '''if w>2048 or h>2048:
        print('---image too large, skip this---')
        continue'''
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
        for name, compression_pipeline in compression_pipelines.items():
            print('current compression: '+name) 
            #print('start time: '+ time.strftime('%d/%m/%Y %H:%M:%S'))
            plaintext_len=img.shape[0]*img.shape[1]*img.shape[2]
            t1=  float("%.9f" % time.time())
            compression_output=compression_pipeline(img)
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
         'experiment':'snu_bwtb64k'
         }
    test_compression(args)
    print('The End.')

