from inception import InceptionV3


class FID():
    def __init__(self, dims):
        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]

        self.inception_model = InceptionV3([block_idx])

    def __call__(generator, num_samples=1000, batch_size=128, dims=2048):
        for i in range(0, num_samples, batch_size):
            img = generator.sample()
            pred = self.inception_model(batch)[0]

            # If model output is not scalar, apply global spatial average pooling.
            # This happens if you choose a dimensionality not equal 2048.
            if pred.size(2) != 1 or pred.size(3) != 1:
                pred = adaptive_avg_pool2d(pred, output_size=(1, 1))

    


