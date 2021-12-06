class LatentDataset:
    def __init__(self, n, init_func=None):
        self.latent = init_func(n)

    def __getitem__(self, index):
        return self.latent[index]

    def __setitem__(self, index, value):
        self.latent[index] = value

    def save(self):
        raise NotImplementedError
