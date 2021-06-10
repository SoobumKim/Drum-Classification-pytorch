import os, shutil
import random
import numpy as np

class Data_train_valid:
    def __init__(self,labels,dataset_path,data_type,time_path,mel_path): # data ratio [v1,aug,v2,v3,v4] train valid ratio [train, valid]
        super(Data_train_valid, self).__init__()
        self.labels = labels
        self.folders = dataset_path
        self.data_types = data_type
        
        self.time_path = time_path
        self.mel_path = mel_path
        
        self.t_folder_1 = self.time_path[0]
        self.t_folder_2 = self.time_path[1]
        self.t_folder_3 = self.time_path[2]
        self.t_folder_4 = self.time_path[3]
        self.t_folder_aug = self.time_path[4]

        self.mel_folder_1 = self.mel_path[0]
        self.mel_folder_2 = self.mel_path[1]
        self.mel_folder_3 = self.mel_path[2]
        self.mel_folder_4 = self.mel_path[3]
        self.mel_folder_aug = self.mel_path[4]
        
    def make_folders(self,status=False):
            try:
                if status == True:
                    for folder in self.folders:
                        for data_type in self.data_types:
                            for label in self.labels:
                                os.makedirs(os.path.join(folder,data_type,label))
                    print("Complete folders generation")
            except:
                print("You already have folders")
                
    def data_copy(self,label,files,data_num,dir_t,dir_mel):
        for n in range(data_num):
            valid_n = int(data_num/np.sum(self.t_v_ratio)*self.t_v_ratio[1])
            if n < valid_n:
                shutil.copy(os.path.join(dir_t,label,files[n]), self.d_dir_t_v)
                shutil.copy(os.path.join(dir_mel,label,files[n]), self.d_dir_mel_v)
            else:
                shutil.copy(os.path.join(dir_t,label,files[n]), self.d_dir_t)
                shutil.copy(os.path.join(dir_mel,label,files[n]), self.d_dir_mel)
                
    def gen_t_v(self,data_info):
        self.total_data_n = data_info['total_data_n'][0]
        self.data_ratio = data_info['data_ratio']
        self.t_v_ratio = data_info['t_v_ratio']
        
        self.data_n = []
        for ratio in self.data_ratio:
            n = int(self.total_data_n/np.sum(self.data_ratio)*ratio)
            self.data_n.append(n)
        
        for label in self.labels: 
            self.d_dir_t = os.path.join(self.folders[0],self.data_types[0],label)
            self.d_dir_mel = os.path.join(self.folders[0],self.data_types[1],label)
            self.d_dir_t_v = os.path.join(self.folders[1],self.data_types[0],label)
            self.d_dir_mel_v = os.path.join(self.folders[1],self.data_types[1],label)
            
            files_1 = os.listdir(self.t_folder_1+label)
            ran_files_1 = random.sample(files_1,len(files_1))
            
            self.data_copy(label,ran_files_1,self.data_n[0],self.t_folder_1,self.mel_folder_1)
            
            files_aug = os.listdir(self.t_folder_aug+label)
            ran_files_aug = random.sample(files_aug,len(files_aug))
            
            self.data_copy(label,ran_files_aug,self.data_n[1],self.t_folder_aug,self.mel_folder_aug)
            
            files_2 = os.listdir(self.t_folder_2+label)
            ran_files_2 = random.sample(files_2,len(files_2))
            
            if len(files_2) > self.data_n[2]:
                self.data_copy(label,ran_files_2,self.data_n[2],self.t_folder_2,self.mel_folder_2)
            else:    
                self.data_copy(label,ran_files_2,len(files_2),self.t_folder_2,self.mel_folder_2)
                
            files_3 = os.listdir(self.t_folder_3+label)
            ran_files_3 = random.sample(files_3,len(files_3))
            
            if len(files_3) > self.data_n[3]:
                self.data_copy(label,ran_files_3,self.data_n[3],self.t_folder_3,self.mel_folder_3)
            else:    
                self.data_copy(label,ran_files_3,len(files_3),self.t_folder_3,self.mel_folder_3)
                
            files_4 = os.listdir(self.t_folder_4+label)
            ran_files_4 = random.sample(files_4,len(files_4))
            
            self.data_copy(label,ran_files_4,self.data_n[4],self.t_folder_4,self.mel_folder_4)
            
            rest_cnt = self.total_data_n - len(os.listdir(self.d_dir_t_v)) - len(os.listdir(self.d_dir_t))
            rest_v = int(self.total_data_n/np.sum(self.t_v_ratio)*self.t_v_ratio[1]) - len(os.listdir(self.d_dir_t_v))
            
            ran_files_aug = ran_files_aug[self.data_n[1]:-1]
            
            for n in range(rest_cnt):
                if n < rest_v:
                    shutil.copy(os.path.join(self.t_folder_aug,label,ran_files_aug[n]), self.d_dir_t_v)
                    shutil.copy(os.path.join(self.mel_folder_aug,label,ran_files_aug[n]), self.d_dir_mel_v)
                else:
                    shutil.copy(os.path.join(self.t_folder_aug,label,ran_files_aug[n]), self.d_dir_t)
                    shutil.copy(os.path.join(self.mel_folder_aug,label,ran_files_aug[n]), self.d_dir_mel)
        
            print("Label- %s : train- %d, valid- %d" %(label,len(os.listdir(self.d_dir_t)),len(os.listdir(self.d_dir_t_v))))