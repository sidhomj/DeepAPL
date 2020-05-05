import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from sklearn.decomposition import NMF
import glob

class ColorNormStains:
    @staticmethod
    def get_background_level(pixel_data, threshold=.8, percentile=0.5):
        # find background as the median of the values above a define threshold
        return np.array([np.quantile(pixel_data[pixel_data[:, i] > threshold, i], percentile) for i in range(pixel_data.shape[-1])])

    @staticmethod
    def histogram_equalization(signal_data, bins, plot_hist=False):
        # get observed density
        h_observed, bin_edges = np.histogram(signal_data, bins, density=True)

        # set target denisty over same bins
        h_target = np.ones_like(h_observed)
        h_target = h_target / (np.diff(bin_edges) * np.sum(h_target))

        # find bin index for each data point
        bin_idx = np.digitize(signal_data, bin_edges, right=False) - 1
        # correct for right of last bin being inclusive
        bin_idx[signal_data == bin_edges[-1]] = len(bin_edges) - 2

        # empirical denisty observed for each data point
        w_observed = h_observed[bin_idx]

        # corrected by the target density
        w_corrective = h_target[bin_idx] / w_observed
        w_corrective = w_corrective / np.sum(w_corrective)

        return w_corrective

    def __init__(self, target_components_od=None):
        if target_components_od is None:
            # these are optical density component matrices - generally these are ones that were solves from a given whole slide image which was felt to be representative
            self.target_components_od = {'Wright-Giemsa': np.array([[2.09723623, 11.99690546, 6.18334853],
                                                                    [6.83504592, 0.68581994, 0.],
                                                                    [0., 0., 6.80148319]])}

    def load_img_files(self, path, extension='.jpg'):
        # load images using glob recursively matching extension
        self.img_names = glob.glob(path + '/**/*' + extension, recursive=True)
        # receiving list of img, this allows from variable size tiles
        img_data = list()
        for img_file in self.img_names:
            img_data.append(cv2.imread(img_file))
        self.process_img_data(img_data)

    def process_img_data(self, img_data):
        # store orginal dimensions of each tile
        self.img_shapes = [img.shape for img in img_data]
        # stack all pixels from all tiles for processing
        self.I_rgb = [img.reshape((-1, img.shape[-1])) for img in img_data]
        # assert that pixels from all tils have the same number of channels
        assert len(np.unique([img.shape[-1] for img in self.I_rgb])) == 1
        self.I_rgb = np.concatenate(self.I_rgb, axis=0)
        # rescale 255 to 1
        if np.max(self.I_rgb) > 1:
            self.I_rgb = self.I_rgb.astype(float) / 255
        # store integrer tile label for each pixel
        self.pixel_tile = list()
        for i in range(len(img_data)):
            self.pixel_tile.append(np.repeat(i, img_data[i].shape[0] * img_data[i].shape[1]))
        self.pixel_tile = np.concatenate(self.pixel_tile)

    def fit(self, n_components=3, n_sampling=250000, init=None):
        # get background intensity values per channel
        I_0 = self.get_background_level(self.I_rgb, threshold=0.8, percentile=0.5)
        self.I_0 = I_0

        # convert from light intensity to optical density and control for background
        self.OD_rgb = -np.log(np.maximum(np.minimum(self.I_rgb / I_0, 1), 1e-3))

        # use total optical density for histogram equalization
        OD_total = np.sum(self.OD_rgb, axis=1)
        corrective_weights = self.histogram_equalization(OD_total, bins=1000)

        # NMF with custom initialization - should be one of the target component matrices
        self.nmf = NMF(n_components=n_components, init='custom')
        self.nmf.get_params()
        # sampling equalized histogram over the distribution of the signal
        idx = np.random.choice(OD_total.shape[0], n_sampling, replace=True, p=corrective_weights)
        self.nmf.fit(self.OD_rgb[idx], W=np.maximum(np.matmul(self.OD_rgb[idx], np.linalg.pinv(self.target_components_od['Wright-Giemsa'])), 0), H=self.target_components_od['Wright-Giemsa'])
        # save the fit optical density component matrices
        self.fit_components = self.nmf.components_

    def transform(self):
        # apply the fitted NMF
        self.OD_nmf = self.nmf.transform(self.OD_rgb)

    def reconstruct_rgb(self, target=None):
        # reconstruct to intensity RGB using fitted optical density NMF and target optical denisty component matrix
        if target in self.target_components_od.keys():
            self.I_rgb_recon = np.exp(-np.matmul(self.OD_nmf, self.target_components_od[target]))
        else:
            print('Warning: target stain not recognized.\nReconstructing using fit stain matrix of these data.')
            self.I_rgb_recon = np.exp(-np.matmul(self.OD_nmf, self.fit_components))

    def channel_plots(self, img):
        # channel plotting function showing each channel and histogram
        n_channels = img.shape[2]
        fig, ax = plt.subplots(nrows=2, ncols=n_channels)
        for i in range(n_channels):
            ax[0, i].imshow(img[:, :, i], cmap='gray')
            ax[0, i].set(xticks=[], yticks=[])
            ax[1, i].hist(img[:, :, i].flatten(), 50)
            ax[1, i].set(yticks=[])

    def Get_Normed_Data(self,nmf=False):
        self.fit()
        self.transform()
        self.reconstruct_rgb('Wright-Giemsa')
        out = [self.I_rgb_recon[self.pixel_tile == i].reshape(self.img_shapes[i]) for i in range(len(self.img_shapes))]
        out_nmf = [self.OD_nmf[self.pixel_tile == i].reshape(self.img_shapes[i]) for i in range(len(self.img_shapes))]
        return out, out_nmf