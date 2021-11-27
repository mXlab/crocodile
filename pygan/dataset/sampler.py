import torch

class SequenceSampler(Sampler):
    def __init__(self, data_source, length, shuffle=True, seed=1234):
        self.data_source = data_source
        self.num_samples = len(self.data_source)
        self.length = length

        self.num_sequences = self.num_samples // self.length

        generator = torch.random.manual_seed(seed)

        self.indices = None
        if not shuffle:
            self.indices = torch.randperm(self.num_sequences, generator=generator).tolist()  

    def __iter__(self):
        list_sequences = torch.arange(self.num_samples).split(self.length)
        indices = self.indices
        if indices is None:
            indices = torch.randperm(len(list_sequences)).tolist()
        
        for i in indices:
            sequence = list_sequences[i]
            if len(sequence) == self.length:
                yield sequence.tolist()
            
    def __len__(self):
        return self.num_sequences