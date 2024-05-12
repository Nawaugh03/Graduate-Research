import numpy as np
import cv2
import scipy.optimize as optimize
from scipy.spatial.transform import Rotation

#https://github.com/sakshikakde/SFM/tree/master
#The following are programs made by the link above 
"""
Linear Pnp
"""
def reprojectionErrorPnP(x3D, pts, K, R, C):
    P = ProjectionMatrix(R,C,K)
    
    Error = []
    for X, pt in zip(x3D, pts):

        p_1T, p_2T, p_3T = P# rows of P
        p_1T, p_2T, p_3T = p_1T.reshape(1,-1), p_2T.reshape(1,-1), p_3T.reshape(1,-1)
        X = homo(X.reshape(1,-1)).reshape(-1,1) # make X it a column of homogenous vector
        ## reprojection error for reference camera points 
        u, v = pt[0], pt[1]
        u_proj = np.divide(p_1T.dot(X) , p_3T.dot(X))
        v_proj =  np.divide(p_2T.dot(X) , p_3T.dot(X))

        E = np.square(v - v_proj) + np.square(u - u_proj)

        Error.append(E)

    mean_error = np.mean(np.array(Error).squeeze())
    return mean_error

def PnP(X_set, x_set, K):
    N = X_set.shape[0]
    
    X_4 = homo(X_set)
    x_3 = homo(x_set)
    
    # normalize x
    K_inv = np.linalg.inv(K)
    x_n = K_inv.dot(x_3.T).T
    
    for i in range(N):
        X = X_4[i].reshape((1, 4))
        zeros = np.zeros((1, 4))
        
        u, v, _ = x_n[i]
        
        u_cross = np.array([[0, -1, v],
                            [1,  0 , -u],
                            [-v, u, 0]])
        X_tilde = np.vstack((np.hstack((   X, zeros, zeros)), 
                            np.hstack((zeros,     X, zeros)), 
                            np.hstack((zeros, zeros,     X))))
        a = u_cross.dot(X_tilde)
        
        if i > 0:
            A = np.vstack((A, a))
        else:
            A = a
            
    _, _, VT = np.linalg.svd(A)
    P = VT[-1].reshape((3, 4))
    R = P[:, :3]
    U_r, D, V_rT = np.linalg.svd(R) # to enforce Orthonormality
    R = U_r.dot(V_rT)
    
    C = P[:, 3]
    C = - np.linalg.inv(R).dot(C)
    
    if np.linalg.det(R) < 0:
        R = -R
        C = -C
        
    return R, C



def project3DPoints(K, R, C, X, image, points):
    I = np.identity(3)
    P = np.dot(K, np.dot(R, np.hstack((I, -C.reshape(3,1)))))
    if X.shape[1] != 4:
        X = homo(X)
    x = np.dot(P, X.T)
    # x01 = np.dot(P1, X01.T)

    x = x/x[2,:]

    xi = x[0, :].T
    yi = x[1, :].T
    im = image.copy()
    for i in range(points.shape[0]):

        x1, y1 = xi[i], yi[i]
        x2, y2 = points[i, 0], points[i, 1]
        cv2.circle(im, (int(x1), int(y1)), 3, (255,0,0), -1)
        cv2.circle(im, (int(x2), int(y2)), 3, (0,255,0), -1)

    cv2.imshow("im", im)
    cv2.waitKey() 
    cv2.destroyAllWindows()
    
        
###################################################### 
########### Reprojection Error Functions #############
######################################################

def ProjectionMatrix(R,C,K):
    C = np.reshape(C, (3, 1))        
    I = np.identity(3)
    P = np.dot(K, np.dot(R, np.hstack((I, -C))))
    return P


def meanReprojectionError(x3D, pts1, pts2, R1, C1, R2, C2, K ):    
    Error = []
    for pt1, pt2, X in zip(pts1, pts2, x3D):
        e1,e2 = ReprojectionError(X, pt1, pt2, R1, C1, R2, C2, K )
        Error.append(e1+e2)
        
    return np.mean(Error)

def ReprojectionError(X, pt1, pt2, R1, C1, R2, C2, K ):
    
    P1 = ProjectionMatrix(R1,C1,K) 
    P2 = ProjectionMatrix(R2,C2,K)

    # X = homo(X.reshape(1,-1)).reshape(-1,1) # make X it a column of homogenous vector
    
    p1_1T, p1_2T, p1_3T = P1 # rows of P1
    p1_1T, p1_2T, p1_3T = p1_1T.reshape(1,-1), p1_2T.reshape(1,-1),p1_3T.reshape(1,-1)

    p2_1T, p2_2T, p2_3T = P2 # rows of P2
    p2_1T, p2_2T, p2_3T = p2_1T.reshape(1,-1), p2_2T.reshape(1,-1), p2_3T.reshape(1,-1)

    ## reprojection error for reference camera points - j = 1
    u1,v1 = pt1[0], pt1[1]
    u1_proj = np.divide(p1_1T.dot(X) , p1_3T.dot(X))
    v1_proj =  np.divide(p1_2T.dot(X) , p1_3T.dot(X))
    E1= np.square(v1 - v1_proj) + np.square(u1 - u1_proj)
    
    ## reprojection error for second camera points - j = 2    
    u2,v2 = pt2[0], pt2[1]
    u2_proj = np.divide(p2_1T.dot(X) , p2_3T.dot(X))
    v2_proj =  np.divide(p2_2T.dot(X) , p2_3T.dot(X))
    
    E2= np.square(v2 - v2_proj) + np.square(u2 - u2_proj)
    
    return E1, E2

def projectPts(R, C, x3D, K):
    I  = np.identity(3)
    P2 = np.dot(K, np.dot(R, np.hstack((I, -C.reshape(3,1)))))
    x3D_4 = homo(x3D)
    x_proj = np.dot(P2, x3D_4.T)
    x_proj = (x_proj/x_proj[2,:]).T
    return x_proj

def homo(pts):
    return np.hstack((pts, np.ones((pts.shape[0], 1))))


########################################################################################################
#Miscellaneous Utilities python script (MiscUtils.py)
def getQuaternion(R2):
    Q = Rotation.from_matrix(R2)
    return Q.as_quat()

def getEuler(R2):
    euler = Rotation.from_matrix(R2)
    return euler.as_rotvec()

def getRotation(Q, type_ = 'q'):
    if type_ == 'q':
        R = Rotation.from_quat(Q)
        return R.as_matrix()
    elif type_ == 'e':
        R = Rotation.from_rotvec(Q)
        return R.as_matrix()

#Estimating Fundamental Matrix.py
def normalize(uv):
    """
    https://www.cc.gatech.edu/classes/AY2016/cs4476_fall/results/proj3/html/sdai30/index.html
    """
    uv_ = np.mean(uv, axis=0)
    u_,v_ = uv_[0], uv_[1]
    u_cap, v_cap = uv[:,0] - u_, uv[:,1] - v_

    s = (2/np.mean(u_cap**2 + v_cap**2))**(0.5)
    T_scale = np.diag([s,s,1])
    T_trans = np.array([[1,0,-u_],[0,1,-v_],[0,0,1]])
    T = T_scale.dot(T_trans)

    x_ = np.column_stack((uv, np.ones(len(uv))))
    x_norm = (T.dot(x_.T)).T

    return  x_norm, T

def EstimateFundamentalMatrix(pts1, pts2):
    normalised = True

    x1,x2 = pts1, pts2

    if x1.shape[0] > 7:
        if normalised == True:
            x1_norm, T1 = normalize(x1)
            x2_norm, T2 = normalize(x2)
        else:
            x1_norm,x2_norm = x1,x2
            
        A = np.zeros((len(x1_norm),9))
        for i in range(0, len(x1_norm)):
            x_1,y_1 = x1_norm[i][0], x1_norm[i][1]
            x_2,y_2 = x2_norm[i][0], x2_norm[i][1]
            A[i] = np.array([x_1*x_2, x_2*y_1, x_2, y_2*x_1, y_2*y_1, y_2, x_1, y_1, 1])

        U, S, VT = np.linalg.svd(A, full_matrices=True)
        F = VT.T[:, -1]
        F = F.reshape(3,3)

        u, s, vt = np.linalg.svd(F)
        s = np.diag(s)
        s[2,2] = 0
        F = np.dot(u, np.dot(s, vt))
        
        if normalised:
            F = np.dot(T2.T, np.dot(F, T1))
            F = F / F[2,2]
        return F

    else:
        return None

#Get Essential matrix from fundamental matrix python script
def getEssentialMatrix(K, F):
    E = K.T.dot(F).dot(K)
    U,s,V = np.linalg.svd(E)
    s = [1,1,0]
    E_ = np.dot(U,np.dot(np.diag(s),V))
    return E_
#ExtractCameraPose.py
def ExtractCameraPose(E):

    """
    Args:
        E (array): Essential Matrix
        K (array): Intrinsic Matrix
    Returns:
        arrays: set of Rotation and Camera Centers
    """

    ##UPDATE
    U, S, V_T = np.linalg.svd(E)
    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])

    # print("E svd U", U)
    # print("E svd S", S)
    # print("E svd U[:, 2]", U[:, 2])
    R = []
    C = []
    R.append(np.dot(U, np.dot(W, V_T)))
    R.append(np.dot(U, np.dot(W, V_T)))
    R.append(np.dot(U, np.dot(W.T, V_T)))
    R.append(np.dot(U, np.dot(W.T, V_T)))
    C.append(U[:, 2])
    C.append(-U[:, 2])
    C.append(U[:, 2])
    C.append(-U[:, 2])

    for i in range(4):
        if (np.linalg.det(R[i]) < 0):
            R[i] = -R[i]
            C[i] = -C[i]

    return R, C

#DisambiguateCameraPose.py
def DisambiguatePose(r_set, c_set, x3D_set):
    best_i = 0
    max_positive_depths = 0
    
    for i in range(len(r_set)):
        R, C = r_set[i],  c_set[i].reshape(-1,1) 
        r3 = R[2, :].reshape(1,-1)
        x3D = x3D_set[i]
        x3D = x3D / x3D[:,3].reshape(-1,1)
        x3D = x3D[:, 0:3]
        n_positive_depths = DepthPositivityConstraint(x3D, r3,C)
        if n_positive_depths > max_positive_depths:
            best_i = i
            max_positive_depths = n_positive_depths
#         print(n_positive_depths, i, best_i)

    R, C, x3D = r_set[best_i], c_set[best_i], x3D_set[best_i]

    return R, C, x3D 

def DepthPositivityConstraint(x3D, r3, C):
    # r3(X-C) alone doesnt solve the check positivity. z = X[2] must also be +ve 
    n_positive_depths=  0
    for X in x3D:
        X = X.reshape(-1,1) 
        if r3.dot(X-C)>0 and X[2]>0: 
            n_positive_depths+=1
    return n_positive_depths

# def DisambiguatePose_cv2(E, x1,x2, K):
#     _, R,t,_ = cv2.recoverPose(E, x1,x2, K)    
#     return R,t

"""Non linear triangulation """
def NonLinearTriangulation(K, pts1, pts2, x3D, R1, C1, R2, C2):
    """    
    K : Camera Matrix
    pts1, pts2 : Point Correspondences
    x3D :  initial 3D point 
    R2, C2 : relative camera pose
    Returns:
        x3D : optimized 3D points
    """
    
    P1 = ProjectionMatrix(R1,C1,K) 
    P2 = ProjectionMatrix(R2,C2,K)
    # pts1, pts2, x3D = pts1, pts2, x3D
    
    if pts1.shape[0] != pts2.shape[0] != x3D.shape[0]:
        raise 'Check point dimensions - level nlt'

    x3D_ = []
    for i in range(len(x3D)):
        optimized_params = optimize.least_squares(fun=ReprojectionLoss, x0=x3D[i], method="trf", args=[pts1[i], pts2[i], P1, P2])
        X1 = optimized_params.x
        x3D_.append(X1)
        # x3D_.append(X1[:3])
    return np.array(x3D_)


def ReprojectionLoss(X, pts1, pts2, P1, P2):
    
    # X = homo(X.reshape(1,-1)).reshape(-1,1) # make X a column of homogenous vector
    
    p1_1T, p1_2T, p1_3T = P1 # rows of P1
    p1_1T, p1_2T, p1_3T = p1_1T.reshape(1,-1), p1_2T.reshape(1,-1),p1_3T.reshape(1,-1)

    p2_1T, p2_2T, p2_3T = P2 # rows of P2
    p2_1T, p2_2T, p2_3T = p2_1T.reshape(1,-1), p2_2T.reshape(1,-1), p2_3T.reshape(1,-1)

    ## reprojection error for reference camera points - j = 1
    u1,v1 = pts1[0], pts1[1]
    u1_proj = np.divide(p1_1T.dot(X) , p1_3T.dot(X))
    v1_proj =  np.divide(p1_2T.dot(X) , p1_3T.dot(X))
    E1= np.square(v1 - v1_proj) + np.square(u1 - u1_proj)

    
    ## reprojection error for second camera points - j = 2    
    u2,v2 = pts2[0], pts2[1]
    u2_proj = np.divide(p2_1T.dot(X) , p2_3T.dot(X))
    v2_proj =  np.divide(p2_2T.dot(X) , p2_3T.dot(X))    
    E2= np.square(v2 - v2_proj) + np.square(u2 - u2_proj)
    
    error = E1 + E2
    return error.squeeze()