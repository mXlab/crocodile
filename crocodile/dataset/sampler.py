import torch
from torch.utils.data import Sampler


class SequenceSampler(Sampler):
    def __init__(self, dataset, length, shuffle=False):
        self.num_samples = len(dataset)
        self.length = length

        self.num_sequences = self.num_samples // self.length

        self.indices = torch.arange(self.num_sequences)
        if shuffle:
            self.indices = torch.randperm(self.num_sequences)

        self.list_sequences =  torch.arange(self.num_samples).split(self.length)

    def __iter__(self): 
        for i in self.indices:
            sequence = self.list_sequences[i]
            yield sequence.tolist()
            
    def __len__(self):
        return self.num_sequences