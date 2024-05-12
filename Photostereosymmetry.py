#Bundle adjustment is a non-linear least squares approach
import random
import cv2 as cv
import pandas as pd
import numpy as np
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
z axis = 1.47m<z<1.6
x axis = ?
y axis = ?

Need at least 8 points to generate a fundamental matrix

f=0.025
camlensXmm=0.46
camlensYmm=0.38


https://docs.opencv.org/3.4/da/de9/tutorial_py_epipolar_geometry.html
"""
Cam2=np.genfromtxt('Cam4raw_coords.csv', delimiter=',')
Cam4=np.genfromtxt('Cam5raw_coords.csv', delimiter=',')

img1=cv.imread('Cam4.avi_frame_0.png',cv.IMREAD_GRAYSCALE)
img2=cv.imread('Cam5.avi_frame_0.png',cv.IMREAD_GRAYSCALE)


LeftCam=Cam2[0]
RightCam=Cam4[0]
#print(LeftCam)

pts1 = []
pts2 = []


Randompts1=[]
Randompts2=[]
Randompts1.append(LeftCam) 
Randompts2.append(RightCam)
for i in range(8):
    Xpts=round(random.uniform((LeftCam[0]-5),LeftCam[0]+5))
    Ypts=round(random.uniform((LeftCam[1]-5),LeftCam[1]+5))
    Randompts1.append((Xpts,Ypts))
    Xpts=round(random.uniform((RightCam[0]-5),RightCam[0]+5))
    Ypts=round(random.uniform((RightCam[1]-5),RightCam[1]+5))
    Randompts2.append((Xpts,Ypts))

print(Randompts1)
print(Randompts2)
pts1 = np.int32(Randompts1)
pts2 = np.int32(Randompts2)

F, mask = cv.findFundamentalMat(pts1,pts2)
print(F)
# We select only inlier points
pts1 = pts1[mask.ravel()==1]
pts2 = pts2[mask.ravel()==1]

def drawlines(img1,img2,lines,pts1,pts2):
    ''' img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''
    r,c = img1.shape
    img1 = cv.cvtColor(img1,cv.COLOR_GRAY2BGR)
    img2 = cv.cvtColor(img2,cv.COLOR_GRAY2BGR)
    for r,pt1,pt2 in zip(lines,pts1,pts2):
        color = tuple(np.random.randint(0,255,3).tolist())
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        img1 = cv.line(img1, (x0,y0), (x1,y1), color,thickness=4)
        img1 = cv.circle(img1,tuple(pt1),5,color,-1)
        img2 = cv.circle(img2,tuple(pt2),5,color,-1)
        
    return img1,img2

# Find epilines corresponding to points in right image (second image) and
# drawing its lines on left image
lines1 = cv.computeCorrespondEpilines(pts2.reshape(-1,1,2), 2,F)
lines1 = lines1.reshape(-1,3)
print(lines1)
img5,img6 = drawlines(img1,img2,lines1,pts1,pts2)

# Find epilines corresponding to points in left image (first image) and
# drawing its lines on right image
lines2 = cv.computeCorrespondEpilines(pts1.reshape(-1,1,2), 1,F)
lines2 = lines2.reshape(-1,3)
img3,img4 = drawlines(img2,img1,lines2,pts2,pts1)
plt.subplot(121),plt.imshow(img5)
plt.subplot(122),plt.imshow(img3)
plt.show()
