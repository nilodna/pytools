import os
import sys
import numpy as np
from scipy.spatial import cKDTree

###############################################################################

def crop_image(fname):
    # function to remove white spaces from pdf and png figures
    extension = fname.split('.')[-1]
    if extension == 'png':
        os.system('convert -trim %s %s'%(fname,fname))
    elif extension == 'pdf':
        os.system('pdfcrop %s %s'%(fname,fname))

###############################################################################

def addpath(path):
    sys.path.append(path)

###############################################################################

def get_uv(int, dir, decl_mag, ang_rot):
	# decompor o vetor
	dir = dir + decl_mag
	dir = np.mod(dir, 360)
	dir = dir * np.pi / 180

	# inlinacao da linha de costa
	# alinhar o sistema com a costa
	ang_rot = ang_rot * np.pi / 180

	u = int * np.sin(dir)	# aqui eu passo de NE para
	v = int * np.cos(dir)	# XY

	U = u * np.cos(ang_rot) - v * np.sin(ang_rot)	# aqui eu faco a rotacao
	V = u * np.sin(ang_rot) + v * np.cos(ang_rot)	# segundo o alinhamento da costa

	return U, V

###############################################################################

def get_intdir(U,V,decl_mag,ang_rot):
    # retorna direcao trigonometrica
	vetor = []
	for i,j in zip(U,V):
		vetor.append(complex(i,j))

	vetor = np.asarray(vetor)

	INT = np.abs(vetor)

	DIR = []
	for d in vetor:
		DIR.append(cmath.phase(d))

	DIR = np.asarray(DIR)
	DIR = DIR * 180 / np.pi # radianos

    # em caso de decl_mag < 0, soma-se para retornar ao zero geográfico (norte)
	DIR = DIR - decl_mag + ang_rot
    # converte do sistema geografico para trigonometrico
	DIR = np.mod(90 - DIR, 360)

	return INT, DIR

###############################################################################
# TODO: WIP - under development yet
def decompose_vectors(u,v,magnetic_declination, rotation_angle_N):
    """
        u,v : wind or current components, referenced to the geographical coordinate system
              (N is 0 and positive is clockwise)
        magnetic_declination : angle of magnetic declination, used when dealing with observational
              dataset. To find the right correction for your dataset, please refer to the [1] and
              then come back to this function.
        rotation_angle_N : angle to rotate the coordinate system. Note that this angle must be provided
              based on the geographical coordinate system. Therefore, provide an angle value starting
              in the north (as the 0 position) and rotating clockwise. The function already convert
              this angle into trigonometric coordinate system during the processing.

        Usage
        -----
        >>> from pytools.sandbox import decompose_vectors
        >>> import matplotlib.pyplot as plt
        >>> u,v = 1,0 # a zonal vector
        >>> angle = 51
        >>> decli_mag = -19
        >>> ur,vr = decompose_vectors(u,v,decli_mag, angle)
        >>> plt.figure()
        >>> plt.quiver(u,v)
        >>> plt.quiver(ur,vr)
    """

    # get intensity and direction from the components
    intensity,direction = get_intdir2(u,v)

    # fix the magnetic declination angle
    direction = direction - magnetic_declination

    # rotate the coordinate system
    direction_geog = direction + rotation_angle_N

    # now, convert this angle to the trigonometric coordinate system
    direction_trig = np.mod(90 - direction_geog, 360)

    # convert back into components, but now we have cross/along shore/isobath
    # but to do that, we use the geographic direction
    ur,vr = get_uv2(u,v,intensity,direction_geog)

    return intensity,direction_geog,ur,vr

def get_uv2(u,v,int,dir):
    ur = +u*np.cos(np.deg2rad(dir)) + v*np.sin(np.deg2rad(dir))
    vr = -u*np.sin(np.deg2rad(dir)) + v*np.cos(np.deg2rad(dir))
    return ur,vr

def get_intdir2(u,v):
    """ Convert velocity from u,v to speed and direction, without any angle correction
    (magnetic declination) or coordinate system rotation. To perform those kind of
    operatioons, please refer to decompose_vectors(u,v), in this same file.

    parameters
    ----------
    u : float
        zonal velocity.
    v : float
        meridional velocity

    returns
    -------
    spd : velocity magnitude
    drt : velocity direction, with 0 pointing to North.

    """

    spd = np.sqrt(u**2 + v**2)
    drt = np.rad2deg(np.arctan2(u,v))

    return spd,drt

###############################################################################

# encontrar indices dos pontos mais proximo a uma coordenada
def find_nearest(lon,lat,ilon,ilat):
    '''
        lon,lat = lat e lon da grade
        ilon,ilat = ponto a ser encontrado
    '''

    lo = lon.ravel()
    la = lat.ravel()

    coords = []

    for i,j in zip(la,lo):
        coords.append([i,j])

    coords = np.array(coords)

    locations_name = ['Terminal Ilha Guaiba']
    locations_posi = [[ilat,ilon]]

    locs = np.asarray(locations_posi)

    tree = cKDTree(coords)
    # procura em tree os pontos mais próximos dos pontos definidos acima
    dists,indexes = tree.query(locs,k=1)

    pontos = []

    for index in indexes:
        pontos.append(coords[index])

    # converter de lista para array
    pontos = np.asarray(pontos)

    # findind indexes from lat and lon
    ind = []

    for p in pontos:
        ind.append(np.where(lon == p[1]))

    ind = np.asarray(ind)

    # vetores para separar i e j para facilitar resgatar os dados de concentração
    iss=[]
    jss=[]

    for i,j in ind:
        iss.append(int(i))
        jss.append(int(j))

    return iss,jss

###############################################################################

def get_datetime(s):
    s = pd.DataFrame(s)
    datetime_list = []
    for i,row in s.iterrows():
        datetime_list.append(parse(row.values[0]))
        
    return np.asarray(datetime_list)

###############################################################################

# function to reduce the linewidth between contourfs
def configuring_for_pdf(cf):

    for c in cf.collections:
        c.set_edgecolor('face')
        c.set_linewidth(0.00000000001)

###############################################################################


def compasstransform(theta):
    '''
    Converts angles between compass direction (clockwise from true North) to direction in polar coordinates (counter-clockwise from x-axis, pointing East).
    Note that regardless of which way the conversion is being done (compass -> polar, or polar -> compass), the output of this function will be the same for a given input.
    INPUT:
    - theta: direction (degrees), numpy array
    OUTPUT:
    - converted direction (degrees), numpy array of same size
        (0 to 360 degrees)
        
    SOURCE: https://github.com/physoce/physoce-py/blob/master/physoce/util.py
    '''
    theta = np.array(theta)
    theta = theta*np.pi/180. # convert to radians
    x = -np.sin(-theta)
    y = np.cos(-theta)
    theta_out = np.arctan2(y,x)
    theta_out = theta_out*180/np.pi # convert back to degrees
    neg = theta_out < 0
    theta_out[neg] = theta_out[neg]+360
    return theta_out

###############################################################################

def matlab2datetime64(datenum,unit='s'):
    '''
    Convert Matlab serial date number to NumPy datetime64 format.
    INPUTS:
    datenum - Matlab serial date number, can be array
    unit - time unit of datetime64 output (default 's')
    OUTPUT:
    array of datetime64 objects
    
    SOURCE: https://github.com/physoce/physoce-py/blob/master/physoce/util.py
    '''
    origin = np.datetime64('0000-01-01 00:00:00', unit) - np.timedelta64(1, 'D')
    daylength = int(np.timedelta64(1,'D')/np.timedelta64(1, unit))
    dt64 = datenum * np.timedelta64(daylength, unit) + origin
    return dt64

###############################################################################