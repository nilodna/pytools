import cmath
import numpy as np
from numba import guvectorize

##############################################################################

@guvectorize(["void(float64[:,:,:], float64[:,:,:], float64[:,:], float64[:,:,:],float64[:,:,:])"],
                "(t,m,n),(t,m,n),(m,n)->(t,m,n), (t,m,n)")
def rot(u,v,angle,urot,vrot):
    """ 
    Note that angle must already converted into radians 
    TODO: adjust function to check the needed to convert angle to radians or not
    """
    for t in range(u.shape[0]):
        for j in range(u.shape[1]):
            for i in range(u.shape[2]):
                ang_rot = angle[j,i] #np.deg2rad(angle[j,i])
                urot[t,j,i] = u[t,j,i] * np.cos(ang_rot) - v[t,j,i] * np.sin(ang_rot)
                vrot[t,j,i] = u[t,j,i] * np.cos(ang_rot) + v[t,j,i] * np.sin(ang_rot)

    return urot,vrot

##############################################################################

@guvectorize(["void(float64[:,:,:], float64[:,:,:], float64[:,:], float64[:,:,:],float64[:,:,:])"],
                "(t,m,n),(t,m,n),(m,n)->(t,m,n), (t,m,n)")
def rot3d(u,v,angle,urot,vrot):
    """ This function applies the decomp collection of routines to rotate velocity components """
    
    # get dimensions
    T,J,Is = u.shape
    
    for t in range(T):
        for j in range(J):
            for i in range(Is):
                ang_rot = (-1)*angle[j,i]
                U, V = u[t, j, i], v[t, j, i]
                I, D = intdir(U, V, 0, ang_rot)
                urot[t, j, i], vrot[t, j, i] = uv(I, D, 0, 0)
    
    return urot,vrot

###############################################################################


def intdir(U,V,decli,rot):
    vetor = np.asarray(complex(U, V))
    
    I = np.abs(vetor)
    D = cmath.phase(vetor)
    
    D = D*180 / np.pi
    D = D - decli + rot
    D = np.mod(90 - D, 360)
    
    return I,D

###############################################################################


def uv(I,D,decli,rot):
    D = D + decli
    D = np.mod(D, 360)
    D = D*np.pi / 180.
    
    rot = rot * np.pi / 180.
    
    u = I * np.sin(D)  # aqui eu passo de NE para
    v = I * np.cos(D)  # XY

    U = u * np.cos(rot) - v * np.sin(rot)  # aqui eu faco a rotacao
    V = u * np.sin(rot) + v * np.cos(rot)
    
    return U,V

###############################################################################
                
