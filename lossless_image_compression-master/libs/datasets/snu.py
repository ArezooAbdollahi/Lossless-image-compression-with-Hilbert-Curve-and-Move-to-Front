import numpy as np
import random
import os.path
from os import listdir
from os.path import isfile, join
import cv2


_root='/media/karimr/hddc/image_compression/code_and_data/dataset/ispl_snu'

class SNU_DataLoader():

    def __init__(self, root=_root):
        self.root=_root
        self.categories=['classic','digital_cam','kodak','medical']
        self.images={ 
                              'classic': [ ],
                              'digital_cam':[],
                              'kodak':[],
                              'medical':[],
                             }
        for cat in self.categories:
            print('cat: '+str(cat))
            path=os.path.join(self.root,cat)
            files = [f for f in listdir(path) if isfile(join(path, f))]
            self.images[cat]=files  
            #print('') 

    def get_categories(self):
        return self.categories

    def get_image_list_by_category(self,category):
        return self.images[category]

    def get_all_data(self):
        return self.images

    def iterator(self):
        for cat,file_list in self.images.items():
            for file_name in file_list:
                full_path=os.path.join(self.root,cat,file_name)
                #print(full_path)   
                img=cv2.imread(full_path,cv2.IMREAD_COLOR)
                #img=img.transpose(2,0,1) #channel last->channel first
                img=img.transpose(2,1,0) #w x h x c->c x h x w 
                yield cat, file_name,img

#--------------------------test------------------------------------------------------#

if __name__=="__main__":
    print('testing ')
    snu=SNU_DataLoader()

