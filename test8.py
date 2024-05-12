#Bundle adjustment is a non-linear least squares approach
import random
import cv2 as cv
import os
import pandas as pd
import numpy as np
import math
from utils import *
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


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

Cam2=np.genfromtxt('Cam2raw_coords.csv', delimiter=',')
Cam4=np.genfromtxt('Cam4raw_coords.csv', delimiter=',')
Cam5=np.genfromtxt('Cam5raw_coords.csv', delimiter=',')
Framenum=0
ImgDirectories=[f"Cam2/Cam2.avi_frame_{Framenum}.png",f"Cam5/Cam5.avi_frame_{Framenum}.png",f"Cam4/Cam4.avi_frame_{Framenum}.png"]
img1=cv.imread(ImgDirectories[0],cv.IMREAD_GRAYSCALE)
img2=cv.imread(ImgDirectories[1],cv.IMREAD_GRAYSCALE)
img3=cv.imread(ImgDirectories[2],cv.IMREAD_GRAYSCALE)
imgsize=img1.shape
print(imgsize)
cx=img1.shape[0]//2
cy=img1.shape[1]//2
#print(cx,cy) #Principal Points
focalLength=25 #25mm
baseline=200 #Changes from 200 to 20 for mm -> m
fx=(cx/460)*25 #Use Later
fy=(cy/380)*25 #Use Later
def CamRotation(pitchX=0,yawY=0,rollZ=0):
    dx=np.radians(pitchX)
    dy=np.radians(yawY)
    dz=np.radians(rollZ)
    Rz =np.array([  [math.cos(dz), -math.sin(dz),  0],
                    [math.sin(dz),  math.cos(dz),  0],
                    [           0,             0,  1]
                    ])
    Ry= np.array([  [math.cos(dy), 0,math.sin(dy)],
                    [0 ,1 ,0],
                    [-math.sin(dy), 0,math.cos(dy)]])
    Rx=np.array([[1, 0 ,0],
                 [0, np.cos(dx), -np.sin(dx)],
                 [0, np.sin(dx),np.cos(dx)]])
    

    R=np.matmul(Rz,np.matmul(Ry,Rx))
    return R
#testRotation=CamRotation(0,0,0)
#print("Cam rotation 0", testRotation)


#                        pitch, yaw, roll3
Cam2_Rotation=CamRotation(0,-10,0) # 90 -10 0
Cam5_Rotation=CamRotation(0,0,0) # 90 0 0
Cam4_Rotation=CamRotation(0,10,0) # 90 10 0
Cam2Coordinates=np.array([[baseline],[0],[0],[1]]) #-baseline
Cam5Coordinates=np.array([[0],[0],[0],[1]])
Cam4Coordinates=np.array([[-baseline],[0],[0],[1]]) #baseline
CamCoordinates=[Cam2Coordinates,Cam5Coordinates,Cam4Coordinates]
Cam2_Translation=np.array([[baseline],[0.0],[0.0]]) #-baseline
Cam5_Translation=np.array([[0.0],[0.0],[0.0]])
Cam4_Translation=np.array([[-baseline],[0.0],[0.0]]) #baseline


index=0
if Framenum==0:
    index=0
else:
    index=Framenum-1
LeftCam=Cam2[index+1]
MiddleCam=Cam5[index+1]
RightCam=Cam4[index+1]
Lpts=np.array([])
Mpts=np.array([])
Rpts=np.array([])
L=np.array([[LeftCam[0]],[LeftCam[1]],[1]])
M=np.array([[MiddleCam[0]],[MiddleCam[1]],[1]])
R=np.array([[RightCam[0]],[RightCam[1]],[1]])

CamMatrixL=np.array([[  fx, 0, cx],
                    [  0,  fy, cy],
                    [  0,   0,  1]])
CamMatrixM=np.array([[  fx, 0,cx],
                    [  0,  fy, cy],
                    [  0,   0,  1]])
CamMatrixR=np.array([[  fx,  0, cx],
                    [  0,  fy, cy],
                    [  0,   0,  1]])
#T=np.concatenate((Cam2_Rotation,Cam2_Translation),axis=1)
#Cam2Pc=np.matmul(T,Cam2Coordinates)
#Cam2P=np.matmul(CamMatrixL,Cam2Pc)

#print(Cam2P/Cam2P[2][0])

dontskip=False
if(dontskip):
    for i in range(1,7):
        Xpts=round(random.uniform((LeftCam[0]-5),LeftCam[0]+5))
        Ypts=round(random.uniform((LeftCam[1]-5),LeftCam[1]+5))
        row=np.array([[Xpts], [Ypts], [1]])
        L=np.concatenate((L, row),axis=1)
        Xpts=round(random.uniform((MiddleCam[0]-5),MiddleCam[0]+5))
        Ypts=round(random.uniform((MiddleCam[1]-5),MiddleCam[1]+5))
        row=np.array([[Xpts], [Ypts], [1]])
        M= np.concatenate((M, row),axis=1)
        Xpts=round(random.uniform((RightCam[0]-5),RightCam[0]+5))
        Ypts=round(random.uniform((RightCam[1]-5),RightCam[1]+5))
        row=np.array([[Xpts], [Ypts], [1]])
        R=np.concatenate((R, row),axis=1)


CamMatrices=[CamMatrixL,CamMatrixM,CamMatrixR]
#Calculate Fundamental Matrix
print()
#Perform the direct linear triangulation between 2 points
def performDLT2Cam(CamMatrixes, LeftPoints, LeftCamRotation, LeftCamTranslation, RightPoints, RightCamRotation, RightCamTranslation):
    LeftCamProjectionoMatrix=CamMatrixes[0]@ np.hstack((LeftCamRotation,LeftCamTranslation))
    RightCamProjectionoMatrix=CamMatrixes [1]@ np.hstack((RightCamRotation,RightCamTranslation))
    #print("Left: ",LeftCamProjectionoMatrix[2,:])
    #print("Right: ",RightCamProjectionoMatrix)
    A=np.zeros((4,4))
    A[0]=LeftPoints[1,:]*LeftCamProjectionoMatrix[2,:] - LeftCamProjectionoMatrix[1,:]
    A[1]=LeftCamProjectionoMatrix[0,:] - LeftPoints[0,:]*LeftCamProjectionoMatrix[2,:]
    A[2]=RightPoints[1,:]*RightCamProjectionoMatrix[2,:] - RightCamProjectionoMatrix[1,:]
    A[3]=RightCamProjectionoMatrix[0,:] - RightPoints[0,:]*RightCamProjectionoMatrix[2,:]
    a,b,Vt=np.linalg.svd(A)
    
    X=Vt[-1]
    X=X/X[-1]
    #return the 3d coordinates of the two values 
    return X[:-1]

#perform the direct linear triangulation between 3 points
def performDLT3Cam(CamMatrices,CamPoints, CamRotations,CamTranslations):
    #CamRotation=np.Cam
    numofpoint=len(CamPoints)
    CamProjections=[]
    for i in range(numofpoint):
        projection=CamMatrices[0]@np.hstack((CamRotations[i],CamTranslations[i]))
        CamProjections.append(projection)
        #print('test',projection,'\n'
    #return CamProjections[2]#CamPoints[i][i,:]#CamProjections[0][0,:]
    A=np.zeros((2*numofpoint, 4))
    for i in range(numofpoint):
        A[2*i]=CamPoints[i][1,:]*CamProjections[i][2,:] - CamProjections[i][1,:]
        A[2*i+1]=CamProjections[i][0,:]-CamPoints[i][0,:]*CamProjections[i][2,:]

    e,f,Vt=np.linalg.svd(A)
    X=Vt[-1]
    X=X/X[3]
    return X
a=performDLT2Cam([CamMatrices[0],CamMatrices[1]],L,Cam2_Rotation,Cam2_Translation,M,Cam5_Rotation,Cam5_Translation)
b=performDLT2Cam([CamMatrices[1],CamMatrices[2]],M,Cam5_Rotation,Cam5_Translation,R,Cam4_Rotation,Cam4_Translation)
c=performDLT2Cam([CamMatrices[2],CamMatrices[0]],L,Cam2_Rotation,Cam2_Translation,R,Cam4_Rotation,Cam4_Translation)

def plot2dSample():
    fig=plt.figure()
    ax=fig.add_subplot()
    images=[plt.imread(ImgDirectories[0]),plt.imread(ImgDirectories[1]),plt.imread(ImgDirectories[2])]
    plt.subplot(1,3,1)
    plt.imshow(images[0])
    plt.scatter(L[0],L[1],color='r')
    plt.title('Cam 2')
    plt.subplot(1,3,2)
    plt.imshow(images[1])
    plt.scatter(M[0],M[1], color='r')
    plt.title('Cam 4')
    plt.subplot(1,3,3)
    plt.imshow(images[2])
    plt.scatter(R[0],R[1], color='r')
    plt.title('Cam 5')
    plt.show()
    
def plot3dSample(points,point3dDLT):
    potentialPoints=np.array(points)
    CamProjections=("Cam2&Cam5" ,"Cam5&Cam4","Cam2&Cam4")
    fig=plt.figure()
    ax = fig.add_subplot(111,projection='3d')
    count=0
    ax.scatter(potentialPoints[:,0],potentialPoints[:,1],potentialPoints[:,2], label="Point", color='r')
    potentialArea=[]
    for i in potentialPoints:
        potentialArea.append(i)
        ax.text(i[0],i[1],i[2],f"{CamProjections[count]}")
        count+=1
    potentialArea.append(potentialPoints[0])
    ax.scatter(point3dDLT[0],point3dDLT[1],point3dDLT[2], label="Point",color='b')
    ax.text(point3dDLT[0],point3dDLT[1],point3dDLT[2], "All 3 points DLT")
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    xl,yl,zl= zip(*potentialArea)
    ax.set_zlim([200,0])
    vert1points=-400
    vert3points=400
    
    # Vertices of the square
    vertices2 = np.array([
        [-380//2, -460//2, 0],
        [380//2,-460//2 , 0],
        [380//2, 460//2, 0],
        [-380//2, 460//2, 0],
        [-380//2, -460//2, 0] 
    ])
    vertices1 = np.array([
        [-380, -460//2, 0],
        [380//2,-460//2 , 0],
        [380//2, 460//2, 0],
        [-380//2, 460//2, 0],
        [-380//2, -460//2, 0] 
    ])
    vertices3 = vertices2
    x1,y1,z1=zip(*vertices1)
    x2,y2,z2=zip(*vertices2)
    x3,y3,z3=zip(*vertices3)
    #=

    # Plot the square surface
    #ax.plot(x1, y1, z1, color='g') 
    ax.plot(x2, y2, z2, color='g',label="Camera 4 field of view")  # Connect the vertices to form the square
    #ax.plot(x3, y3, z3, color='g') 
    ax.view_init(elev=90, azim=270)
    ax.plot(xl,yl,zl, label="Potential Area",color='black')
    ax.plot_trisurf(potentialPoints[:,0],potentialPoints[:,1],potentialPoints[:,2], color='skyblue',alpha=0.5)
    ax.legend(loc='upper left', title='Legend', shadow=True)
    ax.set_box_aspect([1,1,1])
    ax.set_title('3D Graph on where the Rod might be')
    plt.show()

plot2dSample()
print(a)
print(b)
print(c)
#plot3dSample([a,b,c])
Points=[L,M,R]
Rotations=[Cam2_Rotation,Cam5_Rotation,Cam4_Rotation]
Translations=[Cam2_Translation,Cam5_Translation,Cam4_Translation]
d=performDLT3Cam(CamMatrices,Points, Rotations,Translations)
#print(d)
plot3dSample([a,b,c],d)
print(Cam2[0][0])
Coordinates3D=[]
framenum=[]

def Triangulation_3D_3_Cams(x1,y1,x2,y2,x3,y3):
    #CamRotation=np.Cam

    camprojection2=CamMatrixL@np.hstack((Cam2_Rotation,Cam2_Translation)) 
    camprojection4=CamMatrixL@np.hstack((Cam4_Rotation,Cam4_Translation))
    camprojection5=CamMatrixL@np.hstack((Cam5_Rotation,Cam5_Translation))
    A=np.zeros((6, 4))
    A[0]=y1*camprojection2[2,:] - camprojection2[1,:]
    A[1]=camprojection2[0,:]-x1*camprojection2[2,:]
    A[2]=y2*camprojection4[2,:] - camprojection4[1,:]
    A[3]=camprojection4[0,:]-x2*camprojection4[2,:]
    A[4]=y3*camprojection5[2,:] - camprojection5[1,:]
    A[5]=camprojection5[0,:]-x3*camprojection5[2,:]
   

    e,f,Vt=np.linalg.svd(A)
    X=Vt[-1]
    X=X/X[3]
    return X

for i in range(0,300,2):
    if ((Cam2[i][0]!=0 and Cam2[i][0]!=0) or (Cam5[i][0]!=0 and Cam5[i][0]!=0) or (Cam4[i][0]!=0 and Cam4[i][0]!=0)):
        framenum.append(i)
        Coordinates3D.append(Triangulation_3D_3_Cams(Cam2[i][0],Cam2[i][1],Cam4[i][0],Cam4[i][1],Cam5[i][0],Cam5[i][1]))


def plot3dTime(points,framenums):
    #print(points[0])
    fig=plt.figure()
    ax = fig.add_subplot(111,projection='3d')

    for i in range(len(points)):
        ax.scatter(points[i][0],points[i][1],points[i][2], color='r')
        ax.text(points[i][0],points[i][1],points[i][2],f"{framenums[i]}")
    
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    
    

    ax.set_xlim((-460,460))
    ax.set_ylim((-380,380))
    ax.set_zlim((-250,250))
    ax.view_init(elev=90, azim=270)
    #ax.legend(loc='upper left', title='Legend', shadow=True)
    ax.set_box_aspect([1,1,1])
    ax.set_title('3D Graph on where the Rod might be')
    plt.show()

plot3dTime(Coordinates3D,framenum)