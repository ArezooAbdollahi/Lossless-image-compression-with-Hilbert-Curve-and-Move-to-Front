
import numpy as np
import math
from libs.linearization import *
from libs import bwt as bwt
from libs import mtf_coding as mtf
from libs.huffman import *
from libs.metric import *

__bwt_block_size=65536


def pipeline_linearize_bwt_mtf_huffman(data,linearizer=linearize_column_major, use_bwt=False,use_mtf=True):
    encoded=linearizer(data) 
    e1=calculate_entropy(encoded)
    if use_bwt:
        encoded,indexes,split_size=burrows_wheeler_transform(encoded,block_size=__bwt_block_size)          
    if use_mtf:    
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

def column_major_huffman(data):
    return pipeline_linearize_bwt_mtf_huffman(data,linearizer=linearize_column_major, use_bwt=False,use_mtf=False)
	
def column_major_mtf_huffman(data):
    return pipeline_linearize_bwt_mtf_huffman(data,linearizer=linearize_column_major, use_bwt=False,use_mtf=True)

def column_major_burrows_mtf_huffman(data):
    return pipeline_linearize_bwt_mtf_huffman(data,linearizer=linearize_column_major, use_bwt=True,use_mtf=True)
	
def row_major_mtf_huffman(data,use_bwt=False):
    return pipeline_linearize_bwt_mtf_huffman(data,linearizer=linearize_row_major, use_bwt=False,use_mtf=True)

def row_major_burrows_mtf_huffman(data):
    return pipeline_linearize_bwt_mtf_huffman(data,linearizer=linearize_row_major, use_bwt=True,use_mtf=True)

def snakelike_row_major_mtf_huffman(data,use_bwt=False):
    return pipeline_linearize_bwt_mtf_huffman(data,linearizer=linearize_snakelike_row_major, use_bwt=False,use_mtf=True)

def snakelike_row_major_burrows_mtf_huffman(data):
    return pipeline_linearize_bwt_mtf_huffman(data,linearizer=linearize_snakelike_row_major, use_bwt=True,use_mtf=True)
	
def spiral_outer_mtf_huffman(data,use_bwt=False):
    return pipeline_linearize_bwt_mtf_huffman(data,linearizer=linearize_spiral_scan, use_bwt=False,use_mtf=True)

def spiral_outer_burrows_mtf_huffman(data):
    return pipeline_linearize_bwt_mtf_huffman(data,linearizer=linearize_spiral_scan, use_bwt=True,use_mtf=True)

def hilbertcurve_mtf_huffman(data):
    return pipeline_linearize_bwt_mtf_huffman(data,linearizer=linearize_hilbert_curve, use_bwt=False,use_mtf=True)
	
def hilbertcurve_burrows_mtf_huffman(data):
    return pipeline_linearize_bwt_mtf_huffman(data,linearizer=linearize_hilbert_curve, use_bwt=True,use_mtf=True)
	
	


