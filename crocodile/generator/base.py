class Model:
    def __init__(self, model=None, latent_dim=None, device=None):
        self.model = model
        self.device = device
        self.latent_dim = latent_dim

    @staticmethod
    def train(args):
        raise NotImplementedError

    @staticmethod
    def load_model(path: str, device=None):
        raise NotImplementedError

    def sample_z(self, n_samples=1):
        raise NotImplementedError()

    def __call__(self, z):
        # Output image in [-1, 1] with shape BCHW
        raise NotImplementedError()

