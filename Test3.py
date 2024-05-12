#Bundle adjustment is a non-linear least squares approach
import random
import cv2 as cv
import pandas as pd
import numpy as np
import math
from utils import *
from matplotlib import pyplot as plt
"""
https://www.youtube.com/watch?v=sobyKHwgB0Y
A(in frame) Pixel location of i(pointID) in J(camera image) 
                    +
A(in frame) correction discrepancy of i in j

                    =
scaling factor * A(in frame) projectionmatrix in j(image pix, project params distortion(p,q))) * initial guess(3D point)

unknowns
3D locations of new points
1D scale  
6D exterior orientation
5D projection parameters (interriro o.)
Non-linear distortion parameters q

projectionmatrix in j(image pix, project params distortion(p,q)))
                            ||
                            \/
Calibration(interiror oganization)* exterior orientation(Rotation matrix * [I_3| - 3D point of id 0 of camera j])

estimated guess for the center of the ball point
max height = 1.9
z axis = 1.47m<z<1.6
x axis = ?
y axis = ?

Need at least 8 points to generate a fundamental matrix

f=0.025 m
camlensXmm=0.46 m
camlensYmm=0.38 m
x=2056
y=2464


https://docs.opencv.org/3.4/da/de9/tutorial_py_epipolar_geometry.html
"""
Cam2=np.genfromtxt('Cam4raw_coords.csv', delimiter=',')
Cam4=np.genfromtxt('Cam5raw_coords.csv', delimiter=',')
Cam5=np.genfromtxt('Cam5raw_coords.csv', delimiter=',')

img1=cv.imread('Cam4.avi_frame_0.png',cv.IMREAD_GRAYSCALE)
img2=cv.imread('Cam5.avi_frame_0.png',cv.IMREAD_GRAYSCALE)
img3=cv.imread('Cam5.avi_frame_0.png',cv.IMREAD_GRAYSCALE)
x_center=img1.shape[0]//2
y_center=img1.shape[1]//2
CamMatrix=np.zeros((3,3))
CamMatrix[0]=[0.46,0,x_center]
CamMatrix[1]=[0,0.38,y_center]
CamMatrix[2]=[0,0,1]
LeftCam=Cam2[0]
RightCam=Cam4[0]
Cams=[Cam2,Cam4,Cam5]
#print(Cams)
#print(LeftCam)

pts1 = []
pts2 = []
pts1.append(LeftCam)
pts2.append(RightCam)
pts1 = np.int32(pts1)
pts2 = np.int32(pts2)
#print(pts1, pts2)
#print(x_center,y_center)
a=0.46
b=0.38
f=0.025
k=2056/a
l=2464/b

fx=k*f
fy=l*f
#print(fx, fy)
x=pts1[0][0]
y=pts1[0][1]
cot=(1/math.tan(90))
#canera matrices
K=np.array([[fx,0,x_center],
           [0, fy, y_center],
           [0,0,1]])

#print(K)
#Projection matrix= Intrinsic Matrix X Extrinsic Matrix
sampleWorld=np.array([x*a,y*b,1.9])
res=np.dot(sampleWorld,K)
print(res)

