"""
Exercise 1.3 (Different Impacts of Amplitudes and Phase in Frequency Space on
Resulting Filtered Images) It is assumed that you have access to FFT programs for
the 2D DFT and inverse 2D DFT. The task is to study the problem of evaluating
information contained in amplitude and phase of the Fourier transforms:
1. Transform images of identical size into the frequency domain. Map the resulting
complex numbers into amplitudes and phases. Use the amplitudes of one image
and the phases of the other image, and transform the resulting array of complex
numbers back into the spatial domain. Who is “winning”, i.e. can you see the
image contributing the amplitude or the image contributing the phase?
2. Select scalar images showing some type of homogeneous textures; transform
these into the frequency domain and modify either the amplitude or the phase of
the Fourier transform in a uniform way (for all frequencies), before transform-
ing back into the spatial domain. Which modification causes a more significant
change in the image?
3. Do the same operations and tests for a set of images showing faces of human
beings.
Discuss your findings. How do uniform changes (of different degrees), either in
amplitude or in phase, alter the information in the given image
"""

import cv2 
import numpy as np

from math import sqrt, exp, atan2, cos, sin, pi

def ex1():

    img1 =cv2.imread('Images/face1.png', cv2.IMREAD_GRAYSCALE)
    img2 =cv2.imread('Images/face2.png', cv2.IMREAD_GRAYSCALE)

    img1 = cv2.resize(img1, (500,500))
    img2 = cv2.resize(img2, (500,500))


    phase = lambda f_img, w, h: np.array([[ atan2(f_img[j,i].imag, f_img[j,i].real)  for i in range(w)] for j in range(h) ])
    magnitude = lambda f_img, w, h: np.array([[sqrt(f_img[j,i].real**2 + f_img[j,i].imag**2) for i in range(w)] for j in range(h) ])
    to_complex = lambda mag, ph, w, h: np.array([[ complex( mag[j,i]*cos(ph[j,i]) , mag[j,i]*sin(ph[j,i]) ) for i in range(w)] for j in range(h) ])


    img1_dft = np.fft.fft2(img1)
    img1_f_ph = phase(img1_dft, 500, 500)
    img1_f_mag = magnitude(img1_dft, 500, 500)
    
    img2_dft = np.fft.fft2(img2)
    img2_f_ph = phase(img2_dft, 500, 500)
    img2_f_mag = magnitude(img2_dft, 500, 500)


    img3_dft = to_complex(img1_f_mag,img2_f_ph, 500, 500)
    img3 = np.fft.ifft2(img3_dft)
    
    print(img3)
    #print(img.shape, img1_f_mag[0][10])

    cv2.imshow('Img1', img1)
    cv2.imshow('Img2', img2)
    cv2.imshow('Img3', np.real(img3).astype(np.int8))

    key = cv2.waitKey(0)
    if  key & 0xFF == ord('q'):
        return
    
def ex2():
    img1 =cv2.imread('Images/face1.png', cv2.IMREAD_GRAYSCALE)
    img2 =cv2.imread('Images/homo2.png', cv2.IMREAD_GRAYSCALE)

    img1 = cv2.resize(img1, (500,500))
    img2 = cv2.resize(img2, (500,500))


    phase = lambda f_img, w, h: np.array([[ atan2(f_img[j,i].imag, f_img[j,i].real)  + -100 for i in range(w)] for j in range(h) ])
    magnitude = lambda f_img, w, h: np.array([[sqrt(f_img[j,i].real**2 + f_img[j,i].imag**2) + 0 for i in range(w)] for j in range(h) ])
    to_complex = lambda mag, ph, w, h: np.array([[ complex( mag[j,i]*cos(ph[j,i]) , mag[j,i]*sin(ph[j,i]) ) for i in range(w)] for j in range(h) ])


    img1_dft = np.fft.fft2(img1)
    img1_f_ph = phase(img1_dft, 500, 500)
    img1_f_mag = magnitude(img1_dft, 500, 500)

    img3_dft = to_complex(img1_f_mag, img1_f_ph, 500, 500)
    img3 = np.fft.ifft2(img3_dft)
    
    print(img3)
    #print(img.shape, img1_f_mag[0][10])

    cv2.imshow('Img1', img1)
    cv2.imshow('Img3', np.real(img3).astype(np.int8))

    key = cv2.waitKey(0)
    if  key & 0xFF == ord('q'):
        return


if __name__ == '__main__':
    ex2()