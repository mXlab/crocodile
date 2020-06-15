from torch.utils.data import Dataset
import numpy as np
import argparse
import csv



class Emotion(Dataset):
    def __init__(self , label_path, path, sample_length):
        super(Emotion , self).__init__()

        #loads label csv
        self.labels = open(label_path)
        self.labels = csv.reader( self.labels , delimiter = ",")

        #loads biodata signal csv
        self.signal= np.loadtxt(path , delimiter = "," ,dtype={'names': ('heart', 'gsr1', 'gsr2'), 'formats': (float, float, float)})
        
        #time window length in fps @ 30 fps
        self.sample_length = sample_length
        
        #variable used in calculation to align biodata to video
        sampling_rate = 1000,
        fps = 30000/1001
        start_img = 38
        start_data = 300000     
        num_frames = 1000000#find out how many frames in video
        
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
                for i in range( int(start_frame) , int(end_frame)):
                   
                    biodata_index = self.aligned_index[i]
                    biodata_signal_at_frame = self.signal[biodata_index]
                    self.aligned_data.append(biodata_signal_at_frame)
                
                #Create window arrays
                for i in range( n_of_window):
                    start_frame_window = start_frame + i * self.sample_length
                    window = self.aligned_data[start_frame_window : (start_frame_window+self.sample_length)]
                    #print("window length" , len(window))
                    #print(window[0])

                    #store create window in array containing all windows of the same emotion
                    windows.append(window)
                
                #store array of windows corresponding to an emotion in array containing all the windows
                self.data_window.append(windows)





#main loop to test class
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--label_path" , type = str , default = "/Users/etiennemontenegro/Desktop/Crocodile/MNIST_CLASSIFIER/timestamps.csv")
    parser.add_argument("--data_path" , type = str , default = "/Users/etiennemontenegro/Desktop/Crocodile/MNIST_CLASSIFIER/sensor_data.csv")
    parser.add_argument("--sample_length" , type = int , default = 150)
    
    args = parser.parse_args()
    emotion = Emotion(args.label_path, args.data_path, args.sample_length)
    print(len(emotion.aligned_labels))
    print(emotion.aligned_labels)
    print("potato")
    print(len(emotion.data_window[0]))
    #print( emotion.data_window)

