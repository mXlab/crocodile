from torch.utils.data import Dataset
import numpy as np
import argparse
import csv

#ajouter fonction get() retourne element a index specify   et len ( longeur d udataset) (nombre de window)
#ajouter argument train = true pour séparer test et train data
#ajouter un generateur a la classe pour generer tjrs meme numero random
class EmotionDataset(Dataset):
    def __init__(self , label_path, path, sample_length):
        super(EmotionDataset , self).__init__()

        #loads label csv
        self.labels = open(label_path)
        self.labels = csv.reader( self.labels , delimiter = ",")

        #loads biodata signal csv
        #self.signal= np.loadtxt(path , delimiter = "," ,dtype={'names': ('heart', 'gsr1', 'gsr2'), 'formats': (float, float, float)})
        self.signal= np.loadtxt(path , delimiter = "," , dtype=np.float32)

        #time window length in fps @ 30 fps
        self.sample_length = sample_length
        
        #variable used in calculation to align biodata to video
        sampling_rate = 1000,
        fps = 30000/1001
        start_img = 38
        start_data = 300000     
        num_frames = 100000#29147 #find out how many frames in video
        
        #calculation that return array holding the corresponding index of self.signal for the given frame index
        self.aligned_index = (start_data + (np.arange(num_frames) - start_img)*sampling_rate/fps).astype(int)

        #empty arrays that are going to hold aligned data
        self.aligned_data = [] #2D array [ [[heart],[gsr1],[grs2]]
        self.aligned_labels = [] #array that contains label data
        self.data_window = []  # array that contains window data

        #itterate through timestamps.csv rows
        for row in self.labels:
            emotion = row[2]
            
            #conditional statement skipping neutral state
            if emotion != 'Neutral':

                #store emotion label
                self.aligned_labels.append(emotion)
                
                #calculate how many complete window we can make
                start_frame = int(row[0])
                end_frame = int(row[1])
                n_sample = end_frame - start_frame
                n_of_window = int(n_sample / self.sample_length)

                #temporary array that stores windows created for this emotion
                windows = []
               
                #store aligned signal value for every frame in array
                for i in range( int(start_frame) , int(end_frame)-1):
                   
                    biodata_index = self.aligned_index[i]
                    biodata_signal_at_frame = self.signal[biodata_index]
                    self.aligned_data.append(biodata_signal_at_frame)
                
                #Create window arrays
                for i in range( n_of_window):
                    start_frame_window = start_frame + i * self.sample_length
                    window = self.aligned_data[start_frame_window : (start_frame_window+self.sample_length)]
                    #print("window length" , len(window))
                    print('ici' ,window[0])
                    print(type(window[0]))
                    break
                    #store create window in array containing all windows of the same emotion
                    windows.append(window)
                
                #store array of windows corresponding to an emotion in array containing all the windows
                self.data_window.append(windows)


import os
import torch
import torch.nn.functional as F

SAMPLING_RATE = 1000
FPS = 30000/1001
START_IMG = 38
START_DATA = 300000


class EmotionDataset_v2(Dataset):
    def __init__(self, path_to_dataset, path_to_biodata=None, sequence_length=256, normalize=True,
                 permutation=None, split_percent=0.8, train=False, one_hot=False,
                 start_data=START_DATA, start_img=START_IMG, sampling_rate=SAMPLING_RATE, fps=FPS):
        super(EmotionDataset_v2, self).__init__()

        self.one_hot = one_hot
        self.start_data = start_data
        self.start_img = start_img
        self.sampling_rate = sampling_rate
        self.fps = fps

        print("Loading labels...")
        path_to_labels = os.path.join(path_to_dataset, "timestamps.csv")
        if path_to_biodata is None:
            path_to_biodata = os.path.join(path_to_dataset, "LaurenceHBS-Nov919mins1000Hz-Heart+GSR-2channels.csv")

        self.raw_labels = np.loadtxt(path_to_labels, delimiter=",",
                                     dtype={'names': ('start', 'end', 'emotion'), 'formats': (int, int, "<S8")})
        self.labels_to_index = {}
        self.index_to_labels = []
        i = 0
        for row in self.raw_labels:
            if row[2] not in self.labels_to_index:
                self.labels_to_index[row[2]] = i
                self.index_to_labels.append(row[2])
                i += 1
        self.num_cat = len(self.index_to_labels)

        print("Loading biodata...")
        self.signal = torch.tensor(np.loadtxt(path_to_biodata, delimiter=',', usecols=(1, 2)))
        # Split the signal in a list of sequence of size sequence_length   
        list_index = torch.arange(len(self.signal))
        list_sequences = torch.split(list_index, sequence_length)[:-1]
        list_sequences = torch.stack(list_sequences)

        # Create a train/test split
        if permutation is None:
            rng = np.random.default_rng(1234)
            permutation = torch.tensor(rng.permutation(len(list_sequences))).long()
        if train:
            self.dataset = list_sequences[permutation[:int(len(permutation)*split_percent)]]
        else:
            self.dataset = list_sequences[permutation[int(len(permutation)*split_percent):]]
    
        if normalize:
            std, mean = torch.std_mean(self.signal, 0, keepdim=True)
            self.signal = (self.signal - mean) / std

    def __getitem__(self, index):
        # Load sequence
        sequence = self.dataset[index]
        data = self.signal[sequence]
        data = torch.tensor(data).float().transpose(0,1)

        # Load corresponding label
        index_label = (self.start_img + (sequence[-1] - self.start_data)*self.fps/self.sampling_rate).int()
        for row in self.raw_labels:
            if index_label > row[1]:
                continue
            label = self.labels_to_index[row[2]]
            break

        if self.one_hot:
            target = torch.zeros(self.num_cat)
            target[label] = 1
        else:
            target = torch.tensor([label])

        return data, target.long().squeeze()

    def __len__(self):
        return len(self.dataset)


# main loop to test class
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--label_path" , type = str , default = "C:/Users/Etienne/Documents/GitHub/crocodile/MNIST_CLASSIFIER/timestamps.csv")
    parser.add_argument("--data_path" , type = str , default = "C:/Users/Etienne/Documents/GitHub/crocodile/MNIST_CLASSIFIER/sensor_data.csv")
    parser.add_argument("--sample_length" , type = int , default = 150)
    
    args = parser.parse_args()
    emotion = EmotionDataset(args.label_path, args.data_path, args.sample_length)
    print(len(emotion.aligned_labels))
    print(emotion.aligned_labels)
    print("----divider---")
    print(len(emotion.data_window[0]))
    #print( emotion.data_window)

