import torch
from scipy.stats import norm, binom_test
import numpy as np
from math import ceil
from statsmodels.stats.proportion import proportion_confint
import matplotlib.pyplot as plt
from torchvision import transforms

# Adapted from https://github.com/locuslab/smoothing/blob/master/code/core.py
class Smooth(object):
    """A smoothed classifier g """

    # to abstain, Smooth returns this int
    ABSTAIN = -1

    def __init__(self, base_classifier: torch.nn.Module, diffusion_model:torch.nn.Module, sigma: float, t: int):
        """
        :param base_classifier: maps from [batch x channel x height x width] to [batch x num_classes]
        :param num_classes:
        :param sigma: the noise level hyperparameter
        """
        self.base_classifier = base_classifier
        self.diffusion_model = diffusion_model
        self.sigma = sigma
        self.t = t
        self.base_classifier.eval()
        self.diffusion_model.eval()

    def to(self,device):
        self.base_classifier.to(device)
        self.diffusion_model.to(device)
        return self

    def encode_text(self,text_tokens):
        return self.base_classifier.encode_text(text_tokens)

    def encode_image_dif(self, x,num: int = 10, batch_size: int = 1):
        x_batch_zise = x.shape[0]
        sample_size = num
        embeddings_sum = torch.zeros((x_batch_zise, self.base_classifier.visual.output_dim), device=x.device)

        if x.shape[2]<=64:
            for _ in range(ceil(num / batch_size)):
                this_batch_size = min(batch_size, num)
                num -= this_batch_size
                batch = x.repeat((this_batch_size,1, 1, 1))
                # Visualize the images in the batch
                # i=2
                # img = batch[i,:,:,:].permute(1, 2, 0).cpu().numpy()  # Convert from (C, H, W) to (H, W, C)
                # plt.title(f"Image {i + 1}")
                # plt.axis("off")
                # plt.imshow(img)
                # plt.show()
                denoised_imgs = self.diffusion_model(batch, self.t)
                denoised_imgs = (denoised_imgs + 1) / 2  # Rescale to [0, 1] range
                # img = denoised_imgs[i].permute(1, 2, 0).cpu().numpy()  # Convert from (C, H, W) to (H, W, C)
                # plt.title(f"Denoised Image {i + 1}")
                # plt.axis("off")
                # plt.imshow(img)
                # plt.show()
                denoised_imgs = transforms.Normalize(
                    mean=np.array([0.4914, 0.4822, 0.4465]),
                    std=np.array([0.2023, 0.1994, 0.2010])
                )(denoised_imgs)
                embeddings = self.base_classifier.encode_image(denoised_imgs)
                embeddings = embeddings.reshape((this_batch_size,x_batch_zise, -1))
                embeddings_sum += embeddings.sum(dim=0)
        else:
            assert x_batch_zise % batch_size == 0, "Batch size must be a multiple of batch_size for this implementation."
            for _ in range(num):
                for i in range(0, x_batch_zise, batch_size):  # Process in chunks of 5
                    this_batch = x[i:i + batch_size]  # Select a batch of size 5 (or less for the last batch)
                    denoised_imgs = self.diffusion_model(this_batch, self.t)
                    denoised_imgs = (denoised_imgs + 1) / 2  # Rescale to [0, 1] range
                    # img = this_batch[0].permute(1, 2, 0).cpu().numpy()  # Convert from (C, H, W) to (H, W, C)
                    # plt.title(f"Image {i + 1}")
                    # plt.axis("off")
                    # plt.imshow(img)
                    # plt.show()
                    # img = denoised_imgs[0].permute(1, 2, 0).cpu().numpy()  # Convert from (C, H, W) to (H, W, C)
                    # plt.title(f"Denoised Image {i + 1}")
                    # plt.axis("off")
                    # plt.imshow(img)
                    # plt.show()
                    denoised_imgs = transforms.Normalize(
                        mean=np.array([0.485, 0.456, 0.406]),
                        std=np.array([0.229, 0.224, 0.225])
                    )(denoised_imgs)
                    embeddings = self.base_classifier.encode_image(denoised_imgs)
                    embeddings_sum[i:i + this_batch.shape[0], :] += embeddings

        return embeddings_sum / sample_size

    def encode_image(self, x, num: int = 100, batch_size: int = 10):
        x_batch_zise = x.shape[0]
        embeddings_sum = torch.zeros((x_batch_zise, self.base_classifier.visual.output_dim), device=x.device)
        sample_size=num
        for _ in range(ceil(num / batch_size)):
            this_batch_size = min(batch_size, num)
            num -= this_batch_size
            batch = x.repeat((this_batch_size,1, 1, 1))
            noise = torch.randn_like(batch, device=x.device) * self.sigma
            batch = batch + noise
            embeddings = self.base_classifier.encode_image(batch)
            embeddings = embeddings.reshape((this_batch_size,x_batch_zise, -1))
            embeddings_sum += embeddings.sum(dim=0)

        return embeddings_sum / sample_size

    def encode_image_noSmoooth(self, x):
        embeddings = self.base_classifier.encode_image(x)
        return embeddings
