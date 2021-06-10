import torch
from torch.utils.data import Dataset
import numpy as np
import os, glob
import cv2 as cv

class DrumDataset_test(Dataset): #dtype = 'tmel', 'time', 'mel
    def __init__(self,labels,test_path,dtype='tmel'):
        folders = ['images/time','images/mel']
        
        path_t = os.path.join("./",test_path,folders[0])
        path_mel = os.path.join("./",test_path,folders[1])
        
        self.labels = labels
        self.test_data = []
        self.test_label = [] 
        
        if dtype == 'tmel':
            for path_t_name, path_mel_name in zip(glob.glob(os.path.join(path_t,'*')),glob.glob(os.path.join(path_mel,'*'))):
                label = self.label_config(path_t_name)

                img_t = cv.imread(path_t_name,0)  
                img_t = np.expand_dims(img_t, axis=2)
                img_mel = cv.imread(path_mel_name)

                img = np.concatenate((img_t,img_mel),axis=2)
                img = np.transpose(img,(2,0,1))

                self.test_data.append(img)
                self.test_label.append(label) 
            
        elif dtype == 'time':
            for path_t_name in glob.glob(os.path.join(path_t,'*')):
                label = self.label_config(path_t_name)

                img_t = cv.imread(path_t_name,0)  
                img_t = np.expand_dims(img_t, axis=2)
                img = np.transpose(img_t,(2,0,1))

                self.test_data.append(img)
                self.test_label.append(label) 
            
        elif dtype == 'mel':
            for path_mel_name in glob.glob(os.path.join(path_mel,'*')):
                label = self.label_config(path_mel_name)
                
                img_mel = cv.imread(path_mel_name)
                img = np.transpose(img_mel,(2,0,1))

                self.test_data.append(img)
                self.test_label.append(label) 
            
    def label_config(self,path_name):
        data_name = path_name.split('_')
        label_n = data_name[-1][:-4]
        label = self.labels.index(label_n)
        return label
        
    def __len__(self):
        return len(self.test_data)
    
    def __getitem__(self,idx):
        self.test_data = np.array(self.test_data)
        self.test_label = np.array(self.test_label)
        return self.test_data[idx], self.test_label[idx]

