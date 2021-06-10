import librosa, librosa.display
import matplotlib.pyplot as plt
import glob,os
import numpy as np
from scipy.ndimage.interpolation import shift
import warnings
import cv2 as cv
import pandas as pd

class Data_augmentation:
    def __init__(self,labels,root_path='./data/dataset_aug/'): 
        super(Data_augmentation, self).__init__()
        
        self.root_path = root_path
        self.labels = labels
        self.folders = ['images/time','images/mel']
        self.dir_v1 = './data/dataset_v1/audio/'
        
        self.cnt = 0
    
    # Generate aug folders
    def make_aug_folders(self,status=False):
        try:
            if status == True:
                for folder in self.folders:
                    for label in self.labels:
                        os.makedirs(os.path.join(self.root_path,folder,label))
                print("Complete folders generation")
        except:
            print("You already have folders")
            
    # Data Augmentation methods
    def m_Scaling(self,X,f_range_1=0.5, f_range_2=2.0):
        X = np.expand_dims(X,axis=1)
        scalingFactor = np.random.uniform(low=f_range_1, high=f_range_2, size=(1,X.shape[1]))
        myNoise = np.matmul(np.ones((X.shape[0],1)), scalingFactor)
        scaled_X = np.squeeze(X*myNoise,axis=1)
        return scaled_X

    def m_Jitter(self,X,low=0,high=0.01):
        sigma = np.random.uniform(low, high)
        myNoise = np.random.normal(loc=0, scale=sigma, size=X.shape)
        return X+myNoise

    def m_Shifting(self,X,range_1=-300,range_2=1000):
        shift_inv = np.random.randint(range_1,range_2)
        fill_time_noise = np.random.normal(loc=0, scale=0.003, size=abs(shift_inv))
        shift_X = shift(X, shift_inv, cval=np.nan)
        where_nan = np.isnan(shift_X)
        shift_X[where_nan] = fill_time_noise
        return shift_X
    
    def data_aug(self,sound_samples, label, n):
        aug_samples = []
        cnt = 0
        for i in range(n):
            if label == 'OH':
                Sc_sound_samples = self.m_Scaling(sound_samples,0.5,2.0)
                J_sound_samples = self.m_Jitter(Sc_sound_samples,0,0.01)
                ch_sound_samples = J_sound_samples
            elif label == 'CH' or label == 'R':
                Sc_sound_samples = self.m_Scaling(sound_samples,0.2,1.5)
                J_sound_samples = self.m_Jitter(Sc_sound_samples,0,0.03)
                ch_sound_samples = J_sound_samples
            elif label == 'rest':
                Sc_sound_samples = self.m_Scaling(sound_samples,0.5,2.0)
                J_sound_samples = self.m_Jitter(Sc_sound_samples,0,0.01)
                Sh_sound_samples = self.m_Shifting(J_sound_samples,-1000,0)
                ch_sound_samples = Sh_sound_samples
            elif label == 'B+CH' or label == 'B' or label == 'B+OH' or label == 'B+R':
                Sc_sound_samples = self.m_Scaling(sound_samples,1.5,2.5)
                J_sound_samples = self.m_Jitter(Sc_sound_samples,0,0.01)
                Sh_sound_samples = self.m_Shifting(J_sound_samples,-300,1000)
                ch_sound_samples = Sh_sound_samples
            else:
                Sc_sound_samples = self.m_Scaling(sound_samples,0.5,2.0)
                J_sound_samples = self.m_Jitter(Sc_sound_samples,0,0.01)
                Sh_sound_samples = self.m_Shifting(J_sound_samples,-300,1000)
                ch_sound_samples = Sh_sound_samples
            aug_samples.append(ch_sound_samples)
        return aug_samples
    
    def export_data(self,aug_n,label_n):
        label = self.labels[label_n]
        s_dir = os.path.join(self.dir_v1,label)
        for s in os.listdir(s_dir):
            sample_dir = os.path.join(s_dir,s)
            sample, sr = librosa.load(sample_dir)

            aug_samples = self.data_aug(sample, label, aug_n)

            for sample in aug_samples:
                warnings.filterwarnings(action='ignore')
                # time
                plt.figure(figsize=(6.898,4.719))
                plt.plot(sample)
                plt.xlim(0, len(sample))
                plt.ylim(-0.7,0.7)
                plt.axis('off')
                plt.savefig(fname= self.root_path + self.folders[0] + '/'                             + label + '/' + str(self.cnt) + '_aug.jpg', bbox_inches='tight', pad_inches=0)
                plt.close()
                # mel
                plt.figure(figsize=(6.898,4.719))
                f_sample=librosa.feature.melspectrogram(sample)
                librosa.display.specshow(librosa.power_to_db(f_sample), cmap='jet', y_axis='mel')
                plt.clim(-70,20)
                plt.axis('off')
                plt.savefig(fname= self.root_path + self.folders[1] + '/'                             + label + '/' + str(self.cnt) + '_aug.jpg', bbox_inches='tight', pad_inches=0)
                plt.close()

                self.cnt += 1
                    
        print("Complete data augementation")

class Dataset_overview:
    def __init__(self,labels,data_v,root_path='./data'):
        super(Dataset_overview, self).__init__()
        self.root_path = root_path
        self.img_path = 'images/time'
        
        self.labels = labels
        self.folders = data_v
        
    def overview(self):
        data_frame = []
        for folder in self.folders:
            data_cnt = []
            for label in self.labels:
                data_files = glob.glob(os.path.join(self.root_path,folder,self.img_path,label,'*'))
                data_cnt.append(len(data_files))
            data_frame.append(data_cnt)
        df = pd.DataFrame(data_frame,index=self.folders, columns=self.labels)
        df = df.transpose()
        return df
    
    def confirm_size(self):
        error_cnt = 0
        for folder in self.folders:
            for label in self.labels:
                data_files = glob.glob(os.path.join(self.root_path,folder,self.img_path,label,'*'))
                for data_file in data_files:
                    img = cv.imread(data_file)
                    img_size = img.shape
                    if img_size != (256,384,3):
                        error_cnt += 1
        if error_cnt == 0:
            print("All images size is (256,384,3)")
        else:
            print("Number of wrong image size :" + int(error_cnt) )

