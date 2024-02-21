"""
Excercise 1.1

1. Load a colour (RGB) image I in a lossless data format, such as bmp, png, or
    tiff, and display it on a screen.
2. Display the histograms of all three colour channels of I .
3. Move the mouse cursor within your image. For the current pixel location p in the
    image, compute and display
    (a) the outer border (see grey box) of the 11x11 square window Wp around
    pixel p in your image I (i.e., p is the reference point of this window),
    (b) (above this window or in a separate command window) the location p (i.e.,
    its coordinates) of the mouse cursor and the RGB values of your image I at p,
    (c) (below this window or in a separate command window) the intensity value
    [R(p) + G(p) + B(p)]/3 at p, and
    (d) the mean μWp and standard deviation σWp .
4. Discuss examples of image windows Wp (within your selected input images)
    where you see “homogeneous distributions of image values”, and windows showing “inhomogeneous areas”. Try to define your definition of “homogeneous” or
    “inhomogeneous” in terms of histograms, means, or variances.
    The outer border of an 11x11 square window is a 13x13 square curve (which
    could be drawn, e.g., in white) having the recent cursor position at its centre. You
    are expected that you dynamically update this outer border of the 11x11 window
    when moving the cursor
    Alternatively, you could show the 11 × 11 window also in a second frame on
    a screen. Creative thinking is welcome; a modified solution might be even more
    elegant than the way suggested in the text. It is also encouraged to look for solutions
    that are equivalent in performance (same information to the user, similar run time,
    and so forth).
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

from typing import Tuple

G_MAX = 255 # 2^n - 1 -> 2^8 - 1
OUT_BORDER_SIZE = 11

def calculate_histogram(img: np.ndarray, title: str) -> None:
    plt.figure()
    plt.title(title)
    plt.hist(img.ravel(), bins=256, fc='k', ec='k', log=False)

def exercise_1(img_path: str) -> None:
    im = cv2.imread(img_path)
    cv2.imshow("Original", im)
    cv2.waitKey(0)
    
def exercise_2(img_path: str) -> None:
    im = cv2.imread(img_path)
    R = im[:,:,2]
    G = im[:,:,1]
    B = im[:,:,0]
    
    calculate_histogram(R, 'Red')
    calculate_histogram(G, 'Green')
    calculate_histogram(B, 'Blue')
    exercise_1(img_path)

class Coordinates:
    def __init__(self) -> None:
        self.coord = (10,10)
    
    def get_coords(self) -> Tuple[int, int]:
        return self.coord
    
    def update_coords(self, event, x: int, y: int, flags, params) -> None:
        self.coord = (x, y)

def window_mean(img: np.array, p: Tuple[int, int]) -> np.array:
    k =int((OUT_BORDER_SIZE-1)/2)

    mean = [0,0,0]
    for i in range(-k,k):
        for j in range(-k,k):
            mean += img[p[1] + i,p[0] + j,:]
    mean = mean * 1/OUT_BORDER_SIZE**2
    
    print(f'Mean: {mean[2],mean[1],mean[0]}')
    return mean

def window_variance_and_std(img: np.array, p: Tuple[int, int], mean: np.array):
    k =int((OUT_BORDER_SIZE-1)/2)

    var = 0
    for i in range(-k,k):
        for j in range(-k,k):
            var += (img[p[1] + i,p[0] + j,:] - mean)**2
    var = var * 1/OUT_BORDER_SIZE**2

    sd = np.sqrt(var)
    print(f'SD: {sd[2],sd[1],sd[0]}')

def exercise_3(img_path: str) -> None:
    cv2.namedWindow('Original Image')
    cv2.namedWindow('Outer Border')
    
    img = cv2.imread(img_path)
    img = cv2.resize(img, (500,500))
    f,c = img.shape[:2]

    
    outer_border = np.zeros((OUT_BORDER_SIZE, OUT_BORDER_SIZE, 3), np.uint8)
    coords = Coordinates()
    
    cv2.imshow('Original Image', img)
    while True:
        x,y = coords.get_coords()
        
        print(f'Coord: [{x},{y}], RGB: [{img[x,y,2]},{img[x,y,1]},{img[x,y,0]}]')
        print(f'Intensity: {np.sum(img[x,y,:])/3} ')
        
        if (x - OUT_BORDER_SIZE > OUT_BORDER_SIZE and y - OUT_BORDER_SIZE > OUT_BORDER_SIZE) and \
                (x + OUT_BORDER_SIZE < c - OUT_BORDER_SIZE and y + OUT_BORDER_SIZE < f - OUT_BORDER_SIZE):
            
            outer_border = img[y-OUT_BORDER_SIZE:y+OUT_BORDER_SIZE, x-OUT_BORDER_SIZE:x+OUT_BORDER_SIZE]
            mean = window_mean(img, (x,y))
            window_variance_and_std(img, (x,y), mean)
        cv2.imshow('Outer Border', outer_border)

        if cv2.waitKey(60) & 0xFF == ord('q'):
            break
        
        cv2.setMouseCallback('Original Image', coords.update_coords)
    
    
    
if __name__ == '__main__':
    image_path = './Images/Colours.png'
    exercise_3(image_path)