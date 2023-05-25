import scipy
import scipy.misc
import skimage
import skimage.io
import skimage.feature
import skimage.filters
import skimage.morphology
import json

from intersectLines import intersectLines

from skimage.transform import (hough_line, hough_line_peaks,
                               probabilistic_hough_line)
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import copy


def figures4(img1, img2, img3, img4):
    '''
    Plot 4 images in one figure

    :param img1: image 1
    :param img2: image 2
    :param img3: image 3
    :param img4: image 4
    :return:
    '''

    plt.figure(figsize=(9, 4))
    plt.subplot(141)
    plt.title("gray")
    plt.imshow(img1, cmap='gray')
    plt.subplot(142)
    plt.title("red")
    plt.imshow(img2, cmap='gray')
    plt.subplot(143)
    plt.title("diff")
    plt.imshow(img3, cmap='gray')
    plt.subplot(144)
    plt.imshow(img4)

def sob_connect(img, treshold):
    '''
    Tresholding of image from soleb filter

    :param img: image
    :param treshold: treshold
    :return: result image
    '''
    img_connected = img > treshold  # or sob_red[:,:]<0.4
    img_connected = img_connected + (img < -treshold)
    return img_connected

def hugh_lines(img, tested_angles,Viz):
    '''
    Finds lines in image facing given angle

    :param img: image
    :param tested_angles: Angles of lines
    :param Viz: visualisation
    :return: Points on found lines
    '''
    h, theta, d = hough_line(img, theta=tested_angles)
    if Viz: plt.imshow(img, cmap=plt.cm.gray)
    rows, cols = img.shape
    y = []
    for _, angle, dist in zip(*hough_line_peaks(h, theta, d)):
        y0 = (dist - 0 * np.cos(angle)) / np.sin(angle)
        y1 = (dist - cols * np.cos(angle)) / np.sin(angle)
        y.append([y0, y1])
        if Viz: plt.plot((0, cols), (y0, y1), '-r')
    if Viz:  plt.axis((0, cols, rows, 0))
    #plt.title('Detected lines')
    if Viz: plt.show()
    return y

def hugh_prob_lines(lines,Viz):
    '''
    Plot function for line decector

    :param lines: lines
    :param Viz: visualiation
    :return: None
    '''
    # plt.axis([0, imggray.shape[1], imggray.shape[0], 0])
    # plt.gca().set_aspect('auto')
    # # Calculate the desired aspect ratio
    if Viz is False:
        return
    aspect_ratio = imggray.shape[0] / imggray.shape[1]

    # Set the aspect ratio and limits of the axes
    plt.gca().set_aspect(aspect_ratio)

    plt.xlim(0, imggray.shape[1])
    plt.ylim(imggray.shape[0], 0)
    plt.gca().set_aspect('equal', adjustable='box')
    # plt.axis('scaled')
    for line in lines:
        p0, p1 = line
        plt.plot((p0[0], p1[0]), (p0[1], p1[1]))
    plt.show()

def load_file(file_name):
    '''
    Load image of given name from programs folder

    :param file_name: name of file with suffix
    :return: loaded file
    '''
    path='./images/default/'+file_name
    img = skimage.io.imread(path)
    return img

def write_data_to_json_file(name, data):
    '''
    Writes data to JSON file with given name

    :param name: name of file
    :param data: data to write
    :return: None
    '''
    with open(name, 'w') as file:
        json.dump(data, file)


FOLDER_PATH = './images/default/'
files = os.listdir(FOLDER_PATH)
count=0
Viz=True
for file in files:
    count=count+1
    if count>1:
        break
    file_path = FOLDER_PATH + file
    img = load_file(file)
    imggray = skimage.color.rgb2gray(img)
    imgdiff = copy.copy(imggray)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            #imgdiff[i,j] = max(img[i,j,2] - img[i,j,0]*6/10 - img[i,j,1]*4/10,0)
            #imgdiff[i, j] = min(img[i, j, 0]*1/10 + img[i, j, 2] * 1 / 10 + img[i, j, 1] * 8 / 10, 255)
            imgdiff[i,j] = max(int(img[i,j,0])-int(img[i,j,2]*7/10 - int(img[i,j,1]*0/10)),0)
    imgred = img[:,:,2]/255
    imgdiff /= 255
    # figures4(imggray, imgred, imgdiff, img)

    if Viz:
        plt.figure(figsize=(9, 4))
        plt.subplot(141)
        plt.title("gray")
        plt.hist(imggray.ravel(), 40, density=False)
        plt.subplot(142)
        plt.title("red")
        plt.hist(imgred.ravel(), 40, density=False)
        plt.subplot(143)
        plt.title("diff")
        plt.hist(imgdiff.ravel(), 40, density=False)
        plt.subplot(144)
        plt.imshow(img)



    edge_roberts_gray = skimage.filters.roberts(imggray)
    edge_roberts_red = skimage.filters.roberts(imgred)
    edge_roberts_diff = skimage.filters.roberts(imgdiff)
    if Viz:
         figures4(edge_roberts_gray, edge_roberts_red, edge_roberts_diff, img)

    edges_gray = skimage.feature.canny(imggray, sigma=2)
    edges_red = skimage.feature.canny(imgred, sigma=2)
    edges_diff = skimage.feature.canny(imgdiff, sigma=2)

    if Viz:
        figures4(edges_gray, edges_red, edges_diff, img)


    sob_gray = scipy.ndimage.filters.prewitt(imggray, 0)
    sob_red = scipy.ndimage.filters.prewitt(imgred, 0)
    sob_diff = scipy.ndimage.filters.prewitt(imgdiff, 0)

    sob_gray_2 = scipy.ndimage.filters.prewitt(imggray)
    sob_red_2 = scipy.ndimage.filters.prewitt(imgred)
    sob_diff_2 = scipy.ndimage.prewitt(imgdiff)
    if Viz:
        # figures4(sob_gray, sob_red, sob_diff, img)
        figures4(sob_gray_2, sob_red_2, sob_diff_2, img)
    # mostly_red = img[60, 150]
    # mostly_blue = img[40, 25]


    sob_gray_connected = sob_connect(sob_gray, 0.2)
    sob_red_connected = sob_connect(sob_red, 0.2)
    sob_diff_connected = sob_connect(sob_diff, 0.1)

    sob_gray_2_connected = sob_connect(sob_gray_2, 0.1)
    sob_red_2_connected = sob_connect(sob_red_2, 0.1)
    sob_diff_2_connected = sob_connect(sob_diff_2, 0.05)
    if Viz:
        # figures4(sob_gray_connected, sob_red_connected, sob_diff_connected, img)
        body=1

    line_length = 50
    threshold = 20
    line_gap = 10
    lines_gray = probabilistic_hough_line(edges_gray, threshold=threshold, line_length=line_length, line_gap=line_gap)
    lines_red = probabilistic_hough_line(edges_red, threshold=threshold, line_length=line_length, line_gap=line_gap)
    lines_diff = probabilistic_hough_line(edges_diff, threshold=threshold, line_length=line_length, line_gap=line_gap)



    if Viz:
        plt.figure(figsize=(11, 4))
        plt.subplot(141)
        plt.title("gray")
        hugh_prob_lines(lines_gray,Viz)
        plt.subplot(142)
        plt.title("red")
        hugh_prob_lines(lines_red,Viz)
        plt.subplot(143)
        plt.title("diff")
        hugh_prob_lines(lines_diff,Viz)
        plt.subplot(144)
        plt.imshow(img)
    else:
        hugh_prob_lines(lines_diff, Viz)
    # kernel = skimage.morphology.diamond(1)
    # closed_diff = skimage.morphology.binary_erosion(sob_diff_connected, kernel)

    tested_angles = np.linspace(np.pi/2 -np.pi / 3, np.pi/2 + np.pi / 3, 300, endpoint=False)
    if Viz:
        plt.figure(figsize=(9, 4))
        plt.subplot(141)
        plt.title("gray")
        hugh_lines(sob_gray_connected, tested_angles,Viz)
        plt.subplot(142)
        plt.title("red")
        hugh_lines(sob_red_connected, tested_angles,Viz)
        plt.subplot(143)
        plt.title("diff")
        hugh_lines(sob_diff_connected, tested_angles,Viz)
        plt.subplot(144)
        plt.imshow(img)
    else:
        hugh_lines(sob_diff_connected, tested_angles, Viz)

    tested_angles = np.linspace(-np.pi / 12, np.pi / 12, 100, endpoint=False)

    if Viz:
        plt.figure(figsize=(9, 4))
        plt.subplot(141)
        plt.title("gray")
        y = hugh_lines(sob_gray_2_connected, tested_angles,Viz)
        plt.subplot(142)
        plt.title("red")
        y = hugh_lines(sob_red_2_connected, tested_angles,Viz)
        plt.subplot(143)
        plt.title("diff")
        y = hugh_lines(sob_diff_2_connected, tested_angles,Viz)
        plt.subplot(144)
        plt.imshow(img)
    else:
        y = hugh_lines(sob_diff_2_connected, tested_angles,Viz)
    xi, yi, valid, r, s = intersectLines([imggray.shape[0], y[0][0]], [imggray.shape[1], y[0][1]], [imggray.shape[0], y[1][0]], [imggray.shape[1], y[1][1]])
    stredy = []
    indexiky = [None]*len(y)
    for i in range(len(y)):
        for j in range(i+1, len(y)):
            xi, yi, valid, r, s = intersectLines([imggray.shape[0], y[i][0]], [imggray.shape[1], y[i][1]],
                                                 [imggray.shape[0], y[j][0]], [imggray.shape[1], y[j][1]])
            if xi > 0 and xi < imggray.shape[1] and yi > 0 and yi < imggray.shape[0]:
                appended = False
                for k in range(len(stredy)):
                    if abs(stredy[k][0] - yi) < 50 and abs(stredy[k][1] - xi) < 10:
                        appended = True
                        if indexiky[i] is None:
                            indexiky[i] = k

                if not appended:
                    stredy.append([yi, xi])
                    if indexiky[i] is None:
                        indexiky[i] = len(stredy)-1


    # break

print()
# duha = skimage.io.imread('http://pixabay.com/static/uploads/photo/2013/07/12/17/20/luck-152048_640.png')
# plt.figure(figsize=(9, 4))
# plt.subplot(141)
# plt.imshow(duha[:,:,0], cmap='gray')
# plt.subplot(142)
# plt.imshow(duha[:,:,1], cmap='gray')
# plt.subplot(143)
# plt.imshow(duha[:,:,2], cmap='gray')
# plt.subplot(144)
# plt.imshow(duha)