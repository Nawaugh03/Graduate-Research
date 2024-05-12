import numpy as np
import math
focalLength=25 #25mm
imgsize=(2056, 2464)
cx=imgsize[0]//2
cy=imgsize[1]//2
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
CamMatrix=np.array([[  fx, 0, cx],
                    [  0,  fy, cy],
                    [  0,   0,  1]])

#Triangulate the 3 points to get the potential 3d coordinate in the image frames
def Triangulation_3D_3_Cams(x1,y1,x2,y2,x3,y3):
    #CamRotation=np.Cam
    #calcuate the cam projection
    camprojection2=CamMatrix@np.hstack((Cam2_Rotation,Cam2_Translation)) 
    camprojection4=CamMatrix@np.hstack((Cam4_Rotation,Cam4_Translation))
    camprojection5=CamMatrix@np.hstack((Cam5_Rotation,Cam5_Translation))
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
"""
Test case

for i in range(0, 300, 2):
    if ((Cam2[i][0]!=0 and Cam2[i][0]!=0) or (Cam5[i][0]!=0 and Cam5[i][0]!=0) or (Cam4[i][0]!=0 and Cam4[i][0]!=0)):
        framenum.append(i)
        Coordinates3D.append(Triangulation_3D_3_Cams(Cam2[i][0],Cam2[i][1],Cam4[i][0],Cam4[i][1],Cam5[i][0],Cam5[i][1]))

"""