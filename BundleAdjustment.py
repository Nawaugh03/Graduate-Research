from __future__ import print_function
import random
import cv2 as cv
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import urllib.request
import time
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
import glob
import bz2
import os
import numpy as np

"""
Import the data
"""
CamNames=["Cam2","Cam4","Cam5"]
Cam2Points=np.genfromtxt('Cam2raw_coords.csv', delimiter=',')
Cam4Points=np.genfromtxt('Cam4raw_coords.csv', delimiter=',')
Cam5Points=np.genfromtxt('Cam5raw_coords.csv',delimiter=',')
Cam2img=""
Cam4img=""
Cam5img=""
validImages=".png"
currentdirectory=os.getcwd()
for i in CamNames:
    imgs=[]
    newdirectory=os.path.join(currentdirectory, i)
    os.chdir(newdirectory)
    for file in os.listdir(newdirectory):
        ext=os.path.splitext(file)[1]
        if ext.lower() not in validImages:
            continue
        imgs.append(cv.imread(file, cv.IMREAD_GRAYSCALE))
        print(file)
    if("2" in i):
        Cam2img=imgs
    elif("4" in i):
        Cam4img=imgs
    else:
        Cam5img=imgs
    os.chdir(currentdirectory)


print(Cam2img)
print(Cam4img)
print(Cam5img)




