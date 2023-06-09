import scipy
import scipy.misc
import skimage
import skimage.io
import skimage.feature
import skimage.filters
import skimage.morphology
from intersectLines import intersectLines
from skimage.transform import (hough_line, hough_line_peaks,
                               probabilistic_hough_line)
import cv2
from sklearn.linear_model import LinearRegression
import json
import numpy as np
import os
import matplotlib.pyplot as plt
import copy
import sys
import os.path

def figures4(img1, img2, img3, img4):
    '''
    Plot 4 images in one figure

    :param img1: image 1
    :param img2: image 2
    :param img3: image 3
    :param img4: image 4
    :return:
    '''

    plt.figure(figsize=(6, 3))
    plt.subplot(221)
    plt.title("gray")
    plt.imshow(img1, cmap='gray')
    plt.subplot(222)
    plt.title("red")
    plt.imshow(img2, cmap='gray')
    plt.subplot(223)
    plt.title("diff")
    plt.imshow(img3, cmap='gray')
    plt.subplot(224)
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
    #if Viz: plt.show()
    return y

def hugh_prob_lines(lines,Viz,w):
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
    #plt.show()

def line_mean(lines,d,img):
    '''
    Computes mean line for every line group givven

    :param lines: lines
    :param d: 0 is grouping via first coordinate, 1 is grouping via second coordinate
    :param img: image
    :return: mean lines
    '''
    h, w = img.shape
    c=0
    T=[]
    Tl=[]
    if d==0:
        eps=w/30
    else: eps= h/5
    #eps=10
    new_lines=[]
    app=False
    for line in lines:
        app=False
        if len(T)==0:
            T.append(line)
            Tl.append(0)
            continue
        for tr in range(len(T)):
            point1 = line[0]
            point2 = line[1]
            if abs(point1[d]-T[tr][0][d])<eps or abs(point2[d]-T[tr][1][d])<eps:
                Tl.append(tr)
                app=True
                break
        if app is False:
                T.append(line)
                Tl.append(len(T)-1)


    for t in range(len(T)):
        l=0
        p11=0
        p12=0
        p21=0
        p22=0
        for i in range(len(Tl)):
            if Tl[i]==t:
                p11+=lines[i][0][0]
                p12+=lines[i][0][1]
                p21 += lines[i][1][0]
                p22 += lines[i][1][1]
                l+=1
        p11=p11/l
        p12=p12/l
        p21 = p21 / l
        p22 = p22 / l
        new_lines.append([[p11, p12], [p21, p22]])
    return new_lines

def adaptive_sob_connect(img, intensity):
    '''
    Adaptive treshold of edge detection output

    :param img: image
    :param intensity: parametr of intensity
    :return: tresholded image
    '''

    h, w = img.shape
    treshold = 0.3
    while True:
        img_connected = sob_connect(img, treshold)
        if sum(sum(img_connected)) / w / h > intensity:
            return img_connected
        treshold *= 3/4

def eval(inc, stitch):
    '''
    Computes crossing points and angles between incision and stitches

    :param inc: incision lines
    :param stitch: stitches lines
    :return: incision lines, crossing point andcrossing angles
    '''
    cross_pos=[]
    cross_ang=[]
    inc_line=[]
    try:
        for sti in stitch:
            sti=np.array(sti)
            for inci in inc:
                inci=np.array(inci)
                x,y,val,r,s = intersectLines(inci[0], inci[1], sti[0], sti[1])
                if val is False:
                    continue
                else:
                    if inci[0][0]< inci[1][0]:
                        x_inc=inci[0][0]
                    else: x_inc=inci[1][0]
                    x_true=abs(x-x_inc )
                    d1= inci[0]-inci[1]
                    d2=sti[0]-sti[1]
                    temp=np.matmul(d1,np.transpose(d2))
                    n1=np.linalg.norm(d1)
                    n2=np.linalg.norm(d2)
                    ang=np.arccos(temp/(n1*n2))
                    ang=np.rad2deg(ang)
                if val:
                    break
            cross_pos.append(x_true)
            cross_ang.append(ang)
    except:
        cross_pos = []
    inc_line=inc

    return inc_line, cross_pos, cross_ang

def sort_data(x, ang):
    '''
    Sorts x from lovest to highest, while also changing same positions in ang
    :param x: first list
    :param ang: second list
    :return: sorted lists
    '''
    x_sort=[]
    ang_sort=[]
    for i in range(len(x)):
        ind=np.argmin(x)
        x_sort.append(x.pop(ind))
        ang_sort.append(ang.pop(ind))
    return  x_sort, ang_sort

def fix_inc(inc):
    '''
    Linear regresion of lines

    :param inc: lines
    :return: regresion
    '''
    x=[]
    y=[]
    for line in inc:
        x.append(line[0][0])
        x.append(line[1][0])
        y.append(line[0][1])
        y.append(line[1][1])
    x=np.array(x).reshape(-1,1)
    y=np.array(y)
    m=LinearRegression().fit(x,y)
    xmin=min(x)
    xmax=max(x)
    try:
        y1=m.predict(xmin)
        y2=m.predict(xmax)
        new_inc=[[xmin, y1],[xmax,y2]]
    except: new_inc=inc[0]
    return  new_inc

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
    with open(name, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)

# kontrola argumentu
arg=sys.argv[1:]
arglen=len(arg)
if arglen == 0:
    print("No input arguments - starting demo with visualization")
    FOLDER_PATH = './images/default/'
    files = os.listdir(FOLDER_PATH)
    files=files[0:4]
    Viz = True
    data = []
    file_name = "out.json"
else:
    file_name=arg[1]
    if arg[1] == "-v":
        Viz=True
        FOLDER_PATH = './images/default/'
        files=[]
        for i in range(2,arglen):
            files.append(arg[i])
    else:
        Viz = False
        FOLDER_PATH = './images/default/'
        files = []
        for i in range(1, arglen):
            files.append(arg[i])
data=[] #vystup programu
# zpracovavani jednotlivych obrazku
for file in files:
    if os.path.exists(FOLDER_PATH+file) is False and os.path.exists(file) is False:
        print("File " + file + "was not found.")
        continue
    alt = False
    file_path =file
    img = load_file(file)

    imggray = skimage.color.rgb2gray(img)
    h, w = imggray.shape
    if h>w:
        imggray=np.rot90(imggray)
        img=np.rot90(img)
        h, w = imggray.shape

    imgdiff = copy.copy(imggray)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            #imgdiff[i,j] = max(img[i,j,2] - img[i,j,0]*6/10 - img[i,j,1]*4/10,0)
            #imgdiff[i, j] = min(img[i, j, 0]*1/10 + img[i, j, 2] * 1 / 10 + img[i, j, 1] * 8 / 10, 255)
            imgdiff[i,j] = max(int(img[i,j,0])-int(img[i,j,2]*7/10 - int(img[i,j,1]*0/10)),0)
    imgred = img[:,:,2]/255
    imgdiff /= 255
    if Viz:
        figures4(imggray, imgred, imgdiff, img)

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


    sob_gray = scipy.ndimage.prewitt(imggray, 0)
    sob_red = scipy.ndimage.prewitt(imgred, 0)
    sob_diff = scipy.ndimage.prewitt(imgdiff, 0)

    sob_gray_2 = scipy.ndimage.prewitt(imggray)
    sob_red_2 = scipy.ndimage.prewitt(imgred)
    sob_diff_2 = scipy.ndimage.prewitt(imgdiff)
    if Viz:
        # figures4(sob_gray, sob_red, sob_diff, img)
        figures4(sob_gray_2, sob_red_2, sob_diff_2, img)


    intensity = 0.1  # jas v poměru k velikosti obrázku
    sob_gray_connected = adaptive_sob_connect(sob_gray, intensity)
    sob_red_connected = adaptive_sob_connect(sob_red, intensity)
    sob_diff_connected = adaptive_sob_connect(sob_diff, intensity)
    intensity = 0.1# jas v poměru k velikosti obrázku
    sob_gray_2_connected = adaptive_sob_connect(sob_gray_2, intensity)
    sob_red_2_connected = adaptive_sob_connect(sob_red_2, intensity)
    sob_diff_2_connected = adaptive_sob_connect(sob_diff_2, intensity)
    if Viz:
        figures4(sob_gray_connected, sob_red_connected, sob_diff_connected, img)

    tested_angles = np.linspace(np.pi / 2 - np.pi / 10, np.pi / 2 + np.pi / 10, 100, endpoint=False) #uhly ve kterych se hleda linka
    line_length = w
    threshold = 5
    line_gap = 15
    lines_gray = probabilistic_hough_line(edges_gray, threshold=threshold, line_length=line_length, line_gap=line_gap, theta=tested_angles)
    edges = sob_gray_connected
    while line_length>70 and len(lines_gray)<2:
        line_length-=10
        lines_gray = probabilistic_hough_line(edges, threshold=threshold, line_length=line_length,
                                              line_gap=line_gap, theta=tested_angles)
        lines_red = probabilistic_hough_line(edges_red, threshold=threshold, line_length=line_length, line_gap=line_gap)
        lines_diff = probabilistic_hough_line(edges_diff, threshold=threshold, line_length=line_length,
                                              line_gap=line_gap)
        if line_length<70: break
    if len(lines_gray)<1:
        edges= edges_gray
        line_length = 200
        threshold = 5
        line_gap = 15
        while line_length > 70and len(lines_gray)<2:
            line_length -= 10
            lines_gray = probabilistic_hough_line(edges, threshold=threshold, line_length=line_length,
                                                  line_gap=line_gap, theta=tested_angles)
            if line_length < 70: break
    # if len(lines_gray)<1:
    #     lines_gray=hugh_lines(sob_diff_connected, tested_angles, Viz)
    #     #lines_gray=[[0,lines_gray[0][0]], [w,lines_gray[0][1]]]
    #     alt=True




    if Viz:
        plt.figure(figsize=(11, 4))
        plt.subplot(141)
        plt.title("gray")
        hugh_prob_lines(lines_gray,Viz,w)
        plt.subplot(142)
        plt.title("red")
        hugh_prob_lines(lines_red,Viz,w)
        plt.subplot(143)
        plt.title("diff")
        hugh_prob_lines(lines_diff,Viz,w)
        plt.subplot(144)
        plt.imshow(img)
        plt.show()

    # kernel = skimage.morphology.diamond(1)
    # closed_diff = skimage.morphology.binary_erosion(sob_diff_connected, kernel)
    # norm hough
    #tested_angles = np.linspace(np.pi/2 -np.pi / 3, np.pi/2 + np.pi / 3, 300, endpoint=False)
    # if Viz:
    #     plt.figure(figsize=(9, 4))
    #     plt.subplot(141)
    #     plt.title("gray")
    #     jiz=hugh_lines(sob_gray_connected, tested_angles,Viz)
    #     plt.subplot(142)
    #     plt.title("red")
    #     hugh_lines(sob_red_connected, tested_angles,Viz)
    #     plt.subplot(143)
    #     plt.title("diff")
    #     hugh_lines(sob_diff_connected, tested_angles,Viz)
    #     plt.subplot(144)
    #     plt.imshow(img)
    # else:
    #     hugh_lines(sob_diff_connected, tested_angles, Viz)

    tested_angles = np.linspace(-np.pi / 12, np.pi / 12, 100, endpoint=False)

    threshold = 15
    line_length = int(h*7/20) #20
    line_gap = 2
    stitch_diff = probabilistic_hough_line(sob_gray_2_connected, threshold=threshold, line_length=line_length, line_gap=line_gap, theta=tested_angles)
    if Viz:
        plt.figure()
        hugh_prob_lines(stitch_diff, Viz,w)
        plt.imshow(sob_gray_2_connected)
        plt.show()
    st=line_mean(stitch_diff,0,sob_diff_connected)



   # if alt is False:
    jiz=line_mean(lines_gray,1,sob_diff)
    if len(jiz)>=2:
        jiz=fix_inc(jiz)

    # else:
    #     jiz=lines_gray
    #     tested_angles = np.linspace(np.pi / 2 - np.pi / 20, np.pi / 2 + np.pi / 20, 100, endpoint=False)
    #     hugh_lines(edges_gray, tested_angles, True)

    inci,cross, ang=eval(jiz,st)
    cross,ang=sort_data(cross,ang)


    if Viz:
        plt.figure()
        hugh_prob_lines(jiz, Viz, w)
        hugh_prob_lines(st, Viz,w)
        plt.imshow(img)
        plt.show()

    data.append([
        {"filename": "incision001.jpg",
         "incision_polyline": inci,
         "crossing_positions": cross,
         "crossing_angles": ang,
         },
    ])


    # alternativni urceni linek (horsi vysledky)
    # if Viz:
    #     plt.figure(figsize=(9, 4))
    #     plt.subplot(141)
    #     plt.title("gray")
    #     y = hugh_lines(sob_gray_2_connected, tested_angles,Viz)
    #     plt.subplot(142)
    #     plt.title("red")
    #     y = hugh_lines(sob_red_2_connected, tested_angles,Viz)
    #     plt.subplot(143)
    #     plt.title("diff")
    #     y = hugh_lines(sob_diff_2_connected, tested_angles,Viz)
    #     plt.subplot(144)
    #     plt.imshow(img)
    # else:
    #     y = hugh_lines(sob_diff_2_connected, tested_angles,Viz)
    # xi, yi, valid, r, s = intersectLines([imggray.shape[0], y[0][0]], [imggray.shape[1], y[0][1]], [imggray.shape[0], y[1][0]], [imggray.shape[1], y[1][1]])
    # stredy = []
    # indexiky = [None]*len(y)
    # volnyCislicko=0
    # for i in range(len(y)):
    #     for j in range(i+1, len(y)):
    #         xi, yi, valid, r, s = intersectLines([0, y[i][0]], [w, y[i][1]],
    #                                              [0, y[j][0]], [w, y[j][1]])
    #         if xi > 0 and xi < imggray.shape[1] and yi > 0 and yi < imggray.shape[0] and valid:
    #             if indexiky[i] is None and indexiky[j] is None:
    #                 indexiky[i] = volnyCislicko
    #                 indexiky[j] = volnyCislicko
    #                 volnyCislicko += 1
    #             if indexiky[i] is not None:
    #                 indexiky[j] = indexiky[i]
    #             elif indexiky[j] is not None:
    #                 indexiky[i] = indexiky[j]
    # for i in range(len(indexiky)):
    #     if indexiky[i] is None:
    #         indexiky[i]=volnyCislicko
    #         volnyCislicko+=1
    #
    #
    #
    # stitch=[]
    # for i in range(volnyCislicko):
    #     break
    #     lin1=[]
    #     s0=0
    #     s1=0
    #     for ind in range(len(indexiky)):
    #         if indexiky[ind]==i:
    #             lin1.append(y[ind])
    #             s0 = s0 + y[ind][0]
    #             s1 = s1 + y[ind][1]
    #     l=len(lin1)
    #     s0=int(s0/l)
    #     s1=int(s1/l)
    #     stitch.append([[0, s0], [w, s1]])
    #
    # if Viz:
    #     #plt.figure([1,3,3])
    #     y = hugh_lines(sob_diff_2_connected, tested_angles, True)
    #     hugh_prob_lines(stitch, True,w)
    #     plt.imshow(img)
    #     plt.show()

write_data_to_json_file(file_name,data)

