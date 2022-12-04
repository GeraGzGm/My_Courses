import cv2
import matplotlib.pyplot as plt
from mpl_point_clicker import clicker
from matplotlib.widgets import Slider
import numpy as np
import matplotlib
from IPython.display import display, clear_output
from IPython.display import HTML
import os
matplotlib.use('Qt5Cairo')

clear = lambda: os.system('cls')

def draw_contour(img: np.array) -> np.array:
    fig, ax = plt.subplots(constrained_layout=True)
    plt.title('Contour plotting')
    plt.axis('off')
    ax.imshow(img, cmap= 'gray')
    klicker = clicker(ax, ["event"], markers=["o", "x", "*"], colors=["r"])
    plt.show()
    return klicker.get_positions()['event']

def bgr_plot_img(img: np.array, contour: np.array, title: str) -> np.array:
    plt.title(title)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.plot(contour[0], contour[1], '-or')
    plt.grid(False)
    plt.axis('off')
    plt.show()

def plot_img(img: np.array, title: str) -> None:
    plt.title(title)
    plt.imshow(img,cmap='gray')
    plt.grid(False)
    plt.axis('off')
    plt.rcParams['figure.figsize'] = [4, 4]
    plt.show()

def Blur_sliders(img: np.array) -> np.array:
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.title('Blur Image')
    plt.axis('off')

    global blurred_img
    blurred_img = img.copy()

    plt.rcParams["figure.autolayout"] = True
    l = ax.imshow(img, cmap='gray')

    axsize = fig.add_axes([0.25, 0.05, 0.65, 0.03])
    axsigma = fig.add_axes([0.25, 0.01, 0.65, 0.03])

    size = Slider(axsize, 'Kernel size', 1, 99, valinit=3,valstep=2)
    sigma = Slider(axsigma, 'Sigma', 1, 99, valinit=1,valstep=2)

    def update(val):
        global blurred_img
        blurred_img = cv2.GaussianBlur(img, (size.val,size.val), sigmaX = sigma.val)
        l.set_data(blurred_img)
        fig.canvas.draw()

    size.on_changed(update)
    sigma.on_changed(update)
    plt.show()
    return blurred_img

def Blurred_Gradient_Magnitude_Squared(img: np.array) -> np.array:
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    plt.axis('off')
    plt.title('Blurred Squared Magnitude')
    plt.rcParams["figure.autolayout"] = True

    global BGMS_
    BGMS_ = img.copy()

    l = ax.imshow(img, cmap="gray")

    axsize = fig.add_axes([0.25, 0.01, 0.65, 0.03])
    axsigma = fig.add_axes([0.25, 0.07, 0.65, 0.03])
    axsobel = fig.add_axes([0.25, 0.13, 0.65, 0.03])

    sobel = Slider(axsobel, 'Sobel kernel size', 1, 31, valinit=3,valstep=2)
    size = Slider(axsize, 'Gaussian kernel size', 1, 99, valinit=3,valstep=2)
    sigma = Slider(axsigma, 'Sigma', 1, 99, valinit=1,valstep=2)

    def update(val):
        global BGMS_, Squared_Mag
 
        dx = cv2.Sobel(img, cv2.CV_64F, dx = 1, dy = 0, ksize=sobel.val)
        dy = cv2.Sobel(img, cv2.CV_64F, dx = 0, dy = 1, ksize=sobel.val)
        Squared_Mag = np.sqrt(dx**2 + dy**2)**2
        Squared_Mag = cv2.dilate(Squared_Mag,(7,7), iterations = 2)
        BGMS_ = cv2.GaussianBlur(Squared_Mag, (size.val,size.val), sigmaX = sigma.val)
        BGMS_ = (BGMS_ - np.min(BGMS_))/(np.max(BGMS_)-np.min(BGMS_))*255
        l.set_data(BGMS_)
        fig.canvas.draw()

    sobel.on_changed(update)
    size.on_changed(update)
    sigma.on_changed(update)

    plt.show()
    return BGMS_

def image_prepocessing(img: np.array) -> np.array:
    """
    Gaussian blur 
    Calculate Blurred Gradient Magnitude Squared (BGMS)
    """
    blurred_img = Blur_sliders(img)
    BGMS = Blurred_Gradient_Magnitude_Squared(blurred_img)

    return BGMS

def get_Energy(BGMS: np.array, contour: np.array , size: int, bgr_img: np.array) -> None:
    initial_contours = contour.copy()
    #External Energy
    E_image = -BGMS

    gamma = 0.8
    alpha = 0.2
    beta = 0.08

    
    sum_total = []
    f,c = BGMS.shape
    _,points = contour.shape

    for i in range(5):
        for v in range(points):
            minimum = 100000000000
            pos = []
            passed = False

            #
            tot = 0

            if v + 1 < points:
                E_v_elastic = (contour[1][v + 1] - contour[1][v])**2 + (contour[0][v + 1] - contour[1][v])**2
                E_v_smooth = (contour[1][v + 1] - 2*contour[1][v] + contour[1][v - 1])**2 + (contour[0][v + 1] - 2*contour[0][v] + contour[0][v - 1])**2
            else:
                E_v_elastic = (contour[1][0] - contour[1][v])**2 + (contour[0][0] - contour[1][v])**2
                E_v_smooth = (contour[1][0] - 2*contour[1][v] + contour[1][v - 1])**2 + (contour[0][0] - 2*contour[0][v] + contour[0][v - 1])**2

            for i in range(-(size//2), size//2):
                for j in range(-(size//2), size//2):
                    E_v_contour = alpha * E_v_elastic + beta * E_v_smooth

                    if i+contour[1][v] > 0  and i+contour[1][v] < f and j+contour[0][v] > 0  and j+contour[0][v] < c:
                        E_v_image = E_image[i+contour[1][v]][j+contour[0][v]]
                    
                    E_v_total = gamma * E_v_image + E_v_contour
                    if E_v_total < minimum and E_image[i+contour[1][v]][j+contour[0][v]] != E_image[contour[1][v]][contour[0][v]]:
                        minimum = E_v_total 
                        pos = [i+contour[1][v], j+contour[0][v]]
                        passed = True
                        tot = E_v_total

            sum_total.append(tot)
                        
            if passed:
                contour[0][v] = pos[1]
                contour[1][v] = pos[0]
        print(sum_total)
        sum_total = []        
        
    print(sum_total)
    fig = plt.figure()
    plt.title("Initial Contour vs Fixed Contour")
    fig.add_subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.plot(initial_contours[0], initial_contours[1], '-or')
    plt.axis('off')
    fig.add_subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.plot(contour[0], contour[1], '-or')
    plt.axis('off')
    plt.show()


    print(initial_contours.all() == contour.all())
    


if __name__ == "__main__":
    dir_path = "C:/Users/gerag/Desktop/Coursera/Fisrt Principles of Computer Vision/Features and Boundaries/Active Contours GUI/"
    im_name = "car.png"
    bgr_img = cv2.imread(dir_path + im_name)
    img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)

    contour = draw_contour(img).T
    contour = np.hstack((contour, np.array([[contour[0][0]],[contour[1][0]]]))).astype("int16")


    bgr_plot_img(bgr_img, contour, "Contour")
    BGMS = image_prepocessing(img)
    get_Energy(BGMS, contour, 7,bgr_img)
    
    
    
    

    