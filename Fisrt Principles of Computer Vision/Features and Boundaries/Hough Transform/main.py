import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from LassoSelector import SelectFromCollection
from math import sin, cos


def plot_255img(img: np.array, title: str, size=[5, 5]) -> None:
    plt.figure()
    plt.imshow(img, cmap='gray')
    plt.title(title)
    plt.axis("off")
    plt.rcParams['figure.figsize'] = size
    plt.show()


def plot_rgbimg(img: np.array, title: str, size=[5, 5]) -> None:
    plt.figure()
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis("off")
    plt.rcParams['figure.figsize'] = size
    plt.show()


def Edge_detection(img: np.array) -> np.array:
    """
    Canny Edge Detector
    """

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    plt.axis('off')
    plt.title('Blurred Squared Magnitude')

    global img_
    img_ = img.copy()

    l = ax.imshow(img_, cmap="gray")

    axblur = fig.add_axes([0.25, 0.01, 0.65, 0.03])
    axsigma = fig.add_axes([0.25, 0.06, 0.65, 0.03])
    axerode = fig.add_axes([0.25, 0.12, 0.65, 0.03])
    axdilate = fig.add_axes([0.25, 0.18, 0.65, 0.03])

    Blur = Slider(axblur, 'Gaussian kernel size', 1, 90, valinit=3, valstep=2)
    Sigma = Slider(axsigma, 'Gaussian Sigma Param',
                   0.1, 35, valinit=0.1, valstep=0.1)
    Erode = Slider(axerode, 'Erode kernel size', 1, 7, valinit=1, valstep=2)
    Dilate = Slider(axdilate, 'Dilate kernel size', 1, 7, valinit=1, valstep=2)

    def update(val):
        global img_

        blur = cv2.GaussianBlur(img, ksize=(
            Blur.val, Blur.val), sigmaX=Sigma.val)
        edges = cv2.Canny(blur, 10, 80)

        # Remove noise and non-desired lines
        erode = cv2.erode(edges, kernel=np.ones(
            (Erode.val, Erode.val), np.uint8), iterations=1)
        dilate = cv2.dilate(erode, kernel=np.ones(
            (Dilate.val, Dilate.val), np.uint8), iterations=1)
        img_ = dilate
        l.set_data(img_)
        fig.canvas.draw()

    Blur.on_changed(update)
    Sigma.on_changed(update)
    Erode.on_changed(update)
    Dilate.on_changed(update)

    plt.show()

    return img_


def Hough_Transform(edges: np.array) -> np.array:
    f, c = edges.shape
    d = int(np.sqrt(f**2 + c**2))  # Max diatance is diagonal one

    theta = np.arange(0, 180)
    theta_rad = np.deg2rad(theta)

    p_threshold = (-d, d)

    rhos = []

    H = np.zeros((int(2*d), len(theta)+1))  # Hough accumulator

    # Obtain pixels that are edges
    non_zero_x, non_zero_y = np.nonzero(edges)

    for i in range(len(non_zero_x)):
        x = non_zero_x[i]
        y = non_zero_y[i]

        # Calculate rho
        r = np.array(x * np.cos(theta_rad) + y *
                     np.sin(theta_rad), dtype='int')
        rhos.append(np.array(r + d))
        # Increment values
        H[r + d, theta] += 1

    return H, rhos, theta, d


def draw_lines(index, img_, d):

    for idx in index:
        p, theta = idx

        a = sin(np.deg2rad(theta))
        b = cos(np.deg2rad(theta))

        X = (p-d)*a
        Y = (p-d)*b

        x1 = int(X + 1000*(-b))
        y1 = int(Y + 1000*(a))
        x2 = int(X - 1000*(-b))
        y2 = int(Y - 1000*(a))

        img_ = cv2.line(img_, (x1, y1), (x2, y2), (0, 255, 0), 1)
    plt.rcParams['figure.figsize'] = [5, 5]
    plot_rgbimg(img_, "Hough Transform", [5, 5])


def Select_points(H: np.array, rhos: list, theta: np.array):
    global index, count
    count = 0
    index = []

    thetas = [np.array(theta) for _ in range(len(rhos))]
    plt.rcParams['figure.figsize'] = [8, 8]
    subplot_kw = dict(xlim=(np.min(rhos), np.max(rhos)),
                      ylim=(0, 180), autoscale_on=False)
    fig, ax = plt.subplots(subplot_kw=subplot_kw)
    pts = ax.scatter(rhos[::2], thetas[::2], c='k', alpha=0.005)
    selector = SelectFromCollection(ax, pts)

    def accept(event):
        global index, count
        points = []

        if event.key == "enter":
            count += 1
            print("Selected group of points #{}".format(count))
            points = np.array(
                selector.xys[selector.ind], dtype='int').T.tolist()
            max_index = np.argmax(H[points])
            index.append([points[0][max_index], points[1][max_index]])
            fig.canvas.draw()

    fig.canvas.mpl_connect("key_press_event", accept)
    ax.set_title("Press enter to accept selected points.")

    plt.show()

    return index


if __name__ == "__main__":
    dir_path = os.getcwd().replace("\\", "/")+"/Images/"
    im_name = "esq.png"

    bgr_img = cv2.resize(cv2.imread(dir_path+im_name), (300, 300))
    img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)

    edges = Edge_detection(img)
    H, rhos, theta, d = Hough_Transform(edges)

    idx = Select_points(H, rhos, theta)
    draw_lines(idx, bgr_img, d)
