# Drum-Classification-pytorch
## Drum Robot & Algorithm Overview
<img src=https://user-images.githubusercontent.com/19663575/123232659-565e1280-d514-11eb-8619-d1d9fcd7a597.png width="900" height="400">
  
We propose a dataset containing both various drum sounds and multiple drum sounds combined (called the KIST-Drum database1). We extracted 187.5 ms length drum samples from music played only drums and converted them into images. It obtained 28,992 images and increased them to 63,735 images through augmentation. The 23 drum classes corresponding to each image were annotated by file name. To apply a model suitable for the performing robot system, five convolutional neural networks were used and ablation studies were performed by changing the type of input image and the presence of augmented images in the training set.
  
## KIST-DRUM Dataset
<p align="center"><img src=https://user-images.githubusercontent.com/19663575/123533755-f0ab9980-d752-11eb-85e3-9d47da0007b9.png width="800" height="400"></>

We introduce data set of instrumental sample sounds of standard real drum.  Our target classes are bass drum (B), snare drum (S), closed hi-hat (CH), open hi-hat (OH), ride cymbal (R), floor tom (FT) and mid tom (MT). Additionally, we reserve more classes combined sound when beat simultaneously two or more instrument like bass drum and closed hi-hat (B+CH), bass drum and open hi-hat (B+OH), bass drum and ride cymbal (B+R), bass drum and floor tom (B+FT), bass drum and crash cymbal (B+C), snare and bass drum (S+B), snare drum and closed hi-hat (S+CH), snare drum and open hi-hat (S+OH), snare drum and ride cymbal (S+R), snare drum and floor tom (S+FT), snare drum and crash cymbal (S+C), mid and floor tom (MT+FT), snare and bass drum and closed hi-hat (S+B+CH), snare and bass drum and open hi-hat (S+B+OH), snare and bass drum and ride cymbal (S+B+R). There are total of 23 classes including ‘Rest’ when drums don’t onset. 

## Download Drum Samples Dataset
You can download dataset from https://sites.google.com/d/1GQJ5W7wfv_i4ARKIYW7c0aV1KnIt2AVh/p/1RMVQLFuyCXoxI1UiTocWhxVn7UXWpavK/edit

or in Jupyter Notebook,

~~~
import gdown 
google_path = 'https://drive.google.com/uc?id=' 
file_id = '1VsgbtqBhVnCDpXgp9ydW53eBkfDRq7Tf' 
output_name = 'drumdataset.egg' 
gdown.download(google_path+file_id,output_name,quiet=False)
~~~

## Dataset Overview
~~~
from data_augmentation_overview import Dataset_overview
data_overview = Dataset_overview(labels,data_version)
data_overview.overview()
~~~
<img src=https://user-images.githubusercontent.com/19663575/121461034-04cc6880-c9e9-11eb-8483-75e1b9c0502d.JPG width="350" height="400">

## Ablation Test

<img src=https://user-images.githubusercontent.com/19663575/121988568-894a2d00-cdd5-11eb-9982-bd800d01b997.png width="500" height="400"><img src=https://user-images.githubusercontent.com/19663575/122314239-c25bdc00-cf52-11eb-84d3-a8034e239955.png width="300" height="200">

Change input data type (Waveform, Mel-spectrogram, Wavefrom + Mel-spectrogram), presence or absence of augmentation, 5 networks (VGG19, ResNet32, EfficientNetB0, B1, B2).

## Confusion matrix of EfficientNet-B0 + Aug + Mel-spectrogram (TEST)

<p align="center"><img src=https://user-images.githubusercontent.com/19663575/121989057-6bc99300-cdd6-11eb-8ffb-4bcfd5643039.png width="500" height="500"></>

Test dataset consisted of crop images extracted sequentially based on tempo from music played only with real drums not used in training and validation. Like the training set, the size of test dataset is 256x384 and have four channels. The total image data for test are 1334 and included: 375 images of CH; 173 images of B+CH; 74 images of OH; 174 images of S+CH; 41 images of S+OH; 81 images of B+OH; 11 images of B; 29 images of S; 86 images of R; 82 images of B+R; 46 images of S+R; 44 images of B+C; 44 images of Rest; 6 images of MT; 18 images of S+B+CH; 11 images of FT; 8 images of S+FT; 6 images of S+B, 13 images of S+C; 2 images of S+B+R; 2 images of B+FT; 6 images of S+B+OH; and 2 images of MT+FT. Since the classes were extracted from songs, they differ in numbers, and there are relatively fewer classes, such as in S+B+R, B+FT, and MT+FT, which are mainly used as ornaments. 
  
## Cite
