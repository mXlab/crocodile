from .inception import InceptionV3
import torch
from torch.nn.functional import adaptive_avg_pool2d
import numpy as np
from scipy import linalg
from tqdm import tqdm


class FID():
    def __init__(self, path_to_stats=None, dims=2048, device=None):
        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]

        self.inception_model = InceptionV3([block_idx])
        self.inception_model.to(device)
        self.device = device

        if path_to_stats is not None:
            stats = torch.load(path_to_stats)
            self.mu = stats["mu"]
            self.sigma = stats["sigma"]

    def __call__(self, generator, num_samples=1000, batch_size=128):
        with torch.no_grad():
            list_preds = []
            for i in range(0, num_samples, batch_size):
                img = generator.sample(batch_size)
                img = img/2 + 0.5
                
                pred = self.compute_activations(img)

                list_preds.append(pred.cpu())
            list_preds = torch.cat(list_preds).numpy()
            mu = np.mean(list_preds, axis=0)
            sigma = np.cov(list_preds, rowvar=False)

            return self.compute_fid(mu, sigma, self.mu, self.sigma)

    def compute_stats(self, dataloader):
        with torch.no_grad():
            list_preds = []
            for x, _ in tqdm(dataloader):
                x = x.to(self.device)
                x = x/2 + 0.5
                
                pred = self.compute_activations(x)

                list_preds.append(pred.cpu())
            list_preds = torch.cat(list_preds).numpy()
            mu = np.mean(list_preds, axis=0)
            sigma = np.cov(list_preds, rowvar=False)

            print(mu.shape, sigma.shape)
            return mu, sigma
        
    def compute_activations(self, img):
        pred = self.inception_model(img)[0]

        # If model output is not scalar, apply global spatial average pooling.
        # This happens if you choose a dimensionality not equal 2048.
        if pred.size(2) != 1 or pred.size(3) != 1:
            pred = adaptive_avg_pool2d(pred, output_size=(1, 1))
        
        return pred.view(len(pred), -1)

    def compute_fid(self, mu1, sigma1, mu2, sigma2, eps=1e-6):
        """Numpy implementation of the Frechet Distance.
        The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
        and X_2 ~ N(mu_2, C_2) is
                d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
        Stable version by Dougal J. Sutherland.
        Params:
        -- mu1   : Numpy array containing the activations of a layer of the
                inception net (like returned by the function 'get_predictions')
                for generated samples.
        -- mu2   : The sample mean over activations, precalculated on an
                representative data set.
        -- sigma1: The covariance matrix over activations for generated samples.
        -- sigma2: The covariance matrix over activations, precalculated on an
                representative data set.
        Returns:
        --   : The Frechet Distance.
        """

        mu1 = np.atleast_1d(mu1)
        mu2 = np.atleast_1d(mu2)

        sigma1 = np.atleast_2d(sigma1)
        sigma2 = np.atleast_2d(sigma2)

        assert mu1.shape == mu2.shape, \
            'Training and test mean vectors have different lengths'
        assert sigma1.shape == sigma2.shape, \
            'Training and test covariances have different dimensions'

        diff = mu1 - mu2

        # Product might be almost singular
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        if not np.isfinite(covmean).all():
            msg = ('fid calculation produces singular product; '
                'adding %s to diagonal of cov estimates') % eps
            print(msg)
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

        # Numerical error might give slight imaginary component
        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                m = np.max(np.abs(covmean.imag))
                raise ValueError('Imaginary component {}'.format(m))
            covmean = covmean.real

        tr_covmean = np.trace(covmean)

        return (diff.dot(diff) + np.trace(sigma1) +
                np.trace(sigma2) - 2 * tr_covmean)
