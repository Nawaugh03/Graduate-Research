import random
import cv2 as cv
import os
import pandas as pd
import numpy as np
import math
from utils import *
from matplotlib import pyplot as plt
from scipy.optimize import fmin_powell
from mpl_toolkits.mplot3d import Axes3D

Cam2=np.genfromtxt('Cam2raw_coords.csv', delimiter=',')
Cam4=np.genfromtxt('Cam4raw_coords.csv', delimiter=',')
Cam5=np.genfromtxt('Cam5raw_coords.csv', delimiter=',')
Framenum=0
ImgDirectories=[f"Cam2/Cam2.avi_frame_{Framenum}.png",f"Cam4/Cam4.avi_frame_{Framenum}.png",f"Cam5/Cam5.avi_frame_{Framenum}.png"]
img1=cv.imread(ImgDirectories[0],cv.IMREAD_GRAYSCALE)
img2=cv.imread(ImgDirectories[1],cv.IMREAD_GRAYSCALE)
img3=cv.imread(ImgDirectories[2],cv.IMREAD_GRAYSCALE)
imgsize=img1.shape
cx=img1.shape[0]//2
cy=img1.shape[1]//2
#print(cx,cy) #Principal Points
focalLength=25 #25mm
baseline=200
fx=(cx/380)*25
fy=(cy/460)*25
def CamRotation(pitchX=0,yawY=0,rollZ=0):
    if pitchX==0 and yawY==0 and rollZ==0:
        return np.eye(3)
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
    

    R=np.dot(Rz,np.dot(Ry,Rx))
    return R

Cam2_Rotation=CamRotation(90,-10,180)
Cam4_Rotation=CamRotation(90,0,180)
Cam5_Rotation=CamRotation(90,10,180)
Cam2Coordinates=np.array([[-baseline],[0],[0],[1]])
Cam4Coordinates=np.array([[0],[0],[0],[1]])
Cam5Coordinates=np.array([[baseline],[0],[0],[1]])
CamCoordinates=[Cam2Coordinates,Cam4Coordinates,Cam5Coordinates]
Cam2_Translation=np.array([[-baseline],[0.0],[0.0]])
Cam4_Translation=np.array([[0.0],[0.0],[0.0]])
Cam5_Translation=np.array([[baseline],[0.0],[0.0]])
index=0
LeftCam=Cam2[index]
MiddleCam=Cam4[index]
RightCam=Cam5[index]
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

def convert_to_homogenous(pts):
    return np.hstack((pts, np.ones((pts.shape[0], 1))))
def normalize_points_coodrinates(pts,K):
    K_inv=np.linalg.inv(K)
    pts_normalized=pts @ K_inv.T
    return pts_normalized

def eight_point_algorithm(pts1, pts2):
    """
    Compute the Fundamental Matrix F using the Eight-Point Algorithm.

    Args:
        pts1: Array of shape (N, 2) containing pixel coordinates in image 1.
        pts2: Array of shape (N, 2) containing pixel coordinates in image 2.
        K1: Calibration matrix of camera 1 (3x3).
        K2: Calibration matrix of camera 2 (3x3).

    Returns:
        F: The estimated Fundamental Matrix (3x3).
    """
   
    # Construct the A matrix for the linear least squares problem
    totalpts = pts1.shape[0]
    a = np.zeros((totalpts, 9))
    a[:,0] = pts2[:,0]*pts1[:,0]
    a[:,1] = pts2[:,0]*pts1[:,1]
    a[:,2] = pts2[:,0] 
    a[:,3] = pts2[:,1]*pts1[:,0]
    a[:,4] = pts2[:,1]*pts1[:,1]
    a[:,5] = pts2[:,1]
    a[:,6] = pts1[:,0]
    a[:,7] = pts1[:,1]
    a[:,8] = np.ones(totalpts)

    # Perform SVD on A
    _, _, V = np.linalg.svd(a)

    # Extract the fundamental matrix from the last column of V (null space of A)
    F = V.T[:,-1].reshape(3, 3)
    

    return F

def Compute_Essential_Matrix(F, K1,K2):
    # Perform SVD of the Fundamental Matrix F
    E=K2.T@F@K1
    return E
def Relative_poses(EssentialMatrix):
    U,S,Vt=np.linalg.svd(EssentialMatrix)
    # Ensure proper orientation of U and Vt to ensure det(R) = 1
    if np.linalg.det(U) < 0:
        U *= -1
    if np.linalg.det(Vt) < 0:
        Vt *= -1
    W=np.array([[0,-1,0],
                [1,0,0],
                [0,0,1]])
    Ees=U@np.diag([1,1,0])@Vt
    u,s,vt=np.linalg.svd(Ees)
    T1,R1 =(u@W@np.diag([1,1,0]),u@W@vt)
    T2,R2 =(u@W@np.diag([1,1,0]),u@W.T@vt)
    

    E1=np.dot(T1,R1)
    E2=np.dot(T1,R2)
    E3=np.dot(T2,R1)
    E4=np.dot(T2,R2)
    #print((np.linalg.det(R1)))
    #print((np.linalg.det(R2)))
CamMatrices=[CamMatrixL,CamMatrixM,CamMatrixR]
pts1 = []
pts2 = []
Randompts1=[]
Randompts2=[]
Randompts1.append(LeftCam) 
Randompts2.append(MiddleCam)
for i in range(7):
    Xpts=round(random.uniform((LeftCam[0]-5),LeftCam[0]+5))
    Ypts=round(random.uniform((LeftCam[1]-5),LeftCam[1]+5))
    Randompts1.append((Xpts,Ypts))
    Xpts=round(random.uniform((MiddleCam[0]-5),MiddleCam[0]+5))
    Ypts=round(random.uniform((MiddleCam[1]-5),MiddleCam[1]+5))
    Randompts2.append((Xpts,Ypts))


pts1 = np.int32(Randompts1)
pts2 = np.int32(Randompts2)
pts1=convert_to_homogenous(pts1)
pts2=convert_to_homogenous(pts2)
pts1_norm = normalize_points_coodrinates(pts1, CamMatrixL)
pts2_norm = normalize_points_coodrinates(pts2, CamMatrixM)
#print("Pts1Norm",pts1_norm,"\nPts2Norm",pts2_norm)

F=eight_point_algorithm(pts1_norm,pts2_norm)
#print(F)
E=Compute_Essential_Matrix(F,CamMatrixL,CamMatrixM)
#print(P1)
#print(P2)
#print(E)
Relative_poses(E)

def triangulate_point(x1, x2, P1, P2):

    A=np.zeros((4,4))
    print(x2[1,:]*P2[2,:] - P2[1,:])
    A[0]=x1[1,:]*P1[2,:] - P1[1,:]
    A[1]=P1[0,:] - x1[0,:]*P1[2,:]
    A[2]=x2[1,:]*P2[2,:] - P2[1,:]
    A[3]=P2[0,:] - x2[0,:]*P2[2,:]
    a,b,Vt=np.linalg.svd(A)
    
    X=Vt[-1]
    X=X/X[-1]
    #return the 3d coordinates of the two values 
    return X[:-1]



def find_homography(points_source, points_target):
    A  = construct_A(points_source, points_target)
    u, s, vh = np.linalg.svd(A, full_matrices=True)
    
    # Solution to H is the last column of V, or last row of V transpose
    homography = vh[-1].reshape((3,3))
    return homography/homography[2,2]

def construct_A(points_source, points_target):
    assert points_source.shape == points_target.shape, "Shape does not match"
    num_points = points_source.shape[0]

    matrices = []
    for i in range(num_points):
        partial_A = construct_A_partial(points_source[i], points_target[i])
        matrices.append(partial_A)
    return np.concatenate(matrices, axis=0)

def construct_A_partial(point_source, point_target):
    x, y, z = point_source[0], point_source[1], 1
    x_t, y_t, z_t = point_target[0], point_target[1], 1

    A_partial = np.array([
        [0, 0, 0, -z_t*x, -z_t*y, -z_t*z, y_t*x, y_t*y, y_t*z],
        [z_t*x, z_t*y, z_t*z, 0, 0, 0, -x_t*x, -x_t*y, -x_t*z]
    ])
    return A_partial
print(find_homography(pts1,pts2))