"""
Exercise 1.2 (Data Measures on Image Sequences) Define three different data mea-
sures Di (t), i = 1, 2, 3, for analysing image sequences. Your program should do the
following:
1. Read as input an image sequence (e.g. in VGA format) of at least 50 frames.
2. Calculate your data measures Di (t), i = 1, 2, 3, for those frames.
3. Normalize the obtained functions such that all have the same mean and the same
variance.
4. Compare the normalized functions by using the L1 -metric.
Discuss the degree of structural similarity between your measures in dependence of
the chosen input sequence of images.
"""

import cv2
import json
import numpy as np
from glob import glob
import matplotlib.pyplot as plt

from typing import List, Dict


def load_files(folder: str) -> List:
    images = []
    for file in glob(folder + '*.jpg'):
        images.append(np.asarray(cv2.imread(file, cv2.IMREAD_GRAYSCALE)))
    return images

def normalization(images: List, measures: Dict) -> List:
    norm_images = []

    for idx, image in enumerate(images):
        if idx == 0:
            norm_images.append(image)
            continue
        
        alpha = (measures['StandardDeviation'][idx]/measures['StandardDeviation'][0]) * measures['Mean'][0] - measures['Mean'][idx]
        beta = (measures['StandardDeviation'][0]/measures['StandardDeviation'][idx])
        norm = beta * (image + alpha)

        norm_images.append(np.array(norm).astype(np.uint8))
        
    return norm_images

def compare(raw_measures: Dict, norm_measures: Dict):
    L1_metric = {
            'Mean': [],
            'StandardDeviation': [],
            'Contrast': [],
        }
    
    l1_norm = (lambda x,y: abs(x-y)) 

    for i in range(len(raw_measures['Mean'])):
        L1_metric['Mean'].append( l1_norm(raw_measures['Mean'][i],norm_measures['Mean'][i]) )
        L1_metric['StandardDeviation'].append( l1_norm(raw_measures['StandardDeviation'][i],norm_measures['StandardDeviation'][i]) )
        L1_metric['Contrast'].append( l1_norm(raw_measures['Contrast'][i],norm_measures['Contrast'][i]) )

    plt.figure()
    x = np.arange(len(L1_metric['Mean']))

    plt.plot(x, L1_metric['Mean'], color = 'b', label = 'Mean')
    plt.plot(x, L1_metric['StandardDeviation'], color = 'r', label = 'StdDev')
    plt.plot(x, L1_metric['Contrast'], color = 'g', label = 'Contrast')

    plt.xlabel('Image Sequences')
    plt.ylabel('Data Measure Error')
    plt.legend()
    plt.title('L1-metric')

    print(L1_metric['Mean'])

class DataMeasures():
    def __init__(self):
        self.measures = {
            'Mean': [],
            'StandardDeviation': [],
            'Contrast': [],
        }

    @property
    def get_measures(self) -> Dict:
        self.measures['Mean'] = np.array(self.measures['Mean']).astype(float)
        self.measures['StandardDeviation'] = np.array(self.measures['StandardDeviation']).astype(float)
        self.measures['Contrast'] = np.array(self.measures['Contrast']).astype(float)
        return self.measures

    def compute_dm(self, images: List) -> None:
        self.measures['Mean'] = self.mean_dm(images)
        self.measures['StandardDeviation'] = self.sd_dm(images, self.measures['Mean'])
        self.measures['Contrast'] = self.contrast_dm(images)

    def plot_dm(self, title: str) -> None:
        plt.figure()
        x = np.arange(len(self.measures['Mean']))

        plt.plot(x, self.measures['Mean'], color = 'b', label = 'Mean')
        plt.plot(x, self.measures['StandardDeviation'], color = 'r', label = 'StdDev')
        plt.plot(x, self.measures['Contrast'], color = 'g', label = 'Contrast')

        plt.xlabel('Image Sequences')
        plt.ylabel('Data Measure Values')
        plt.legend()
        plt.title(f'{title}')

    def mean_dm(self, images: List) -> List:
        mean_array = []

        for image in images:
            h, w = image.shape

            mean = 1/(h*w) * np.sum(np.sum(image))
            mean_array.append(mean)

        return mean_array    

    def sd_dm(self, images: List, mean: List) -> np.array:
        variance_array = []

        for idx, image in enumerate(images):
            h, w = image.shape

            im2 = np.square(image.copy().astype(np.int32))
            variance = ( 1/(h*w) * np.sum(im2)  ) - mean[idx]**2

            variance_array.append(variance)

        sd = np.sqrt(variance_array)
        
        return sd  

    def contrast_dm(self, images: List) -> List:
        contrast_array = []

        for image in images:
            h, w = image.shape
            mean_img = np.zeros_like(image)

            for i in range(w):
                for j in range(h):
                    neighs = [image[nj,ni] for ni in range(i - 1 , i + 2) for nj in range(j - 1 , j + 2) if (ni >= 0 and nj >= 0) and (ni < w and nj < h) and (ni != i or nj != j)]

                    mean_neigh = np.mean(neighs)
                    mean_img[j,i] = mean_neigh

            contrast = np.sum(abs(image -  mean_img))
            contrast *= 1/(h * w)
            contrast_array.append(contrast)
        
        return contrast_array

def main():
    images = load_files('./Images/marple2/')

    raw_dm = DataMeasures()
    norm_dm = DataMeasures()

    raw_dm.compute_dm(images)
    raw_measures = raw_dm.get_measures

    norm_images = normalization(images, raw_measures)
    norm_dm.compute_dm(norm_images)
    norm_measures = norm_dm.get_measures

    raw_dm.plot_dm('Raw Sequence')
    norm_dm.plot_dm('Normalized Sequence')
    
    compare(raw_measures, norm_measures)

    plt.show()
    run = True
    while run:
        for raw_image, norm_image in zip(images, norm_images):
            cv2.imshow('Raw', raw_image)
            cv2.imshow('Norm', norm_image)


            key = cv2.waitKey(60)
            if  key & 0xFF == ord('q'):
                run = False
                break

if __name__ == '__main__':
    main()