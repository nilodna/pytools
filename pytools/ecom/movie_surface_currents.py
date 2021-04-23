#!/bin/env python3
"""
    This script generates an animated movie of surface currents based on a POM-Rain output given as argument.
    The final video has normalized vectors indicating the direction of the current, in the oceanographic 
    convention, and the magnitude is presented as colors.
    
    Usage
    -----
    >> $ python movie_surface_currents.py full/path/to/output.cdf name_output_movie_without_extension
    
    This will save a .mp4 file whenever you call the function.
"""
import xarray as xr
import matplotlib.pyplot as plt
from dateutil import parser
import pandas as pd
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import sys, os
import cmocean as cmo

from xmovie import Movie

#---| using ecompy xgcmplt method to read an output |---#
from ecompy.io.xgcmplt import xgcmplt

sys.path.append('/home/danilo/Research/PhD/repos/pytools')
from pytools.sandbox.rotate_angles_numba import rot3d

##############################################################################


def xquiver_format(ds):
    fname = '../data/index_list.npy'
    j, i = np.load(fname, allow_pickle=True)
    # just because we already remove some points during the reading process with
    # xgcmplt
    i = i[:-4]
    j = j[:-2]

    return ds.isel(ypos=i, xpos=j)

##############################################################################

#---------| auxiliar function to compute the intensity from xarray |---------#
def magnitude(u, v):
    def func(x, y): return np.sqrt(u**2 + v**2)

    return xr.apply_ufunc(func, u, v)

##############################################################################

#-----------| customized function to create a movie using xMovie |------------#
def custom_quiver(ds, fig, tt):

    sub = ds.isel(time=tt)
    # subsampling the vectors density for a better visualization
    sub_quiver = xquiver_format(sub)

    ax = fig.add_subplot(111, projection=ccrs.PlateCarree())

    # normalizing vector to the same lenght. The magnitude is given as colors
    unorm, vnorm = sub_quiver.u/sub_quiver.speed, sub_quiver.v/sub_quiver.speed

    sub.speed.plot.pcolormesh(x='lon', y='lat', vmin=0, vmax=.8)
    
    plt.quiver(sub_quiver.lon.values, sub_quiver.lat.values, 
               unorm.values, vnorm.values, 
               scale=60, width=0.0015, headwidth=4, headlength=4)
    
    sub.depth.plot.contour(x='lon', y='lat', levels=[200], colors=('w'), linewidths=(.3))

    land_10m = cfeature.NaturalEarthFeature('physical', 'land', '10m',
                                            edgecolor='face',
                                            facecolor=cfeature.COLORS['land'])
    ax.add_feature(land_10m)
    ax.coastlines('10m')

    ax.set_title(pd.to_datetime(sub.time.values).strftime('%Y-%m-%d %H'))

    fig.subplots_adjust(wspace=0.6)

##############################################################################

def main(argv):
    fname = sys.argv[1]
    fout = sys.argv[2]
    rotation = False
    
    file_model_grid = '/mnt/media/Danilo/Research/PhD/ECOM/ecompy/data/model_grid_withLandPoints'
    
    tmp = xgcmplt(file_model_grid, fname, nrows=26)
    ds = tmp.ds.copy()

    del tmp

    #--| clean xarray.Dataset for faster processing |--#
    #--|     and select only surface currents       |--#
    variables = ['u', 'v', 'depth', 'ang']
    ds = ds[variables].isel(sigma=slice(0, 2)).mean(dim='sigma')

    #--| removing outter borders |--#
    ds = ds.isel(ypos=slice(2, -2), xpos=slice(2, -2))

    #-- calculating magnitude --#
    ds['speed'] = magnitude(ds.u, ds.v)
    
    #-- rotating vectors --#
    if rotation:
        urot = np.zeros(ds.u.shape)*np.nan
        vrot = np.zeros(ds.u.shape)*np.nan
        urot, vrot = rot3d(ds.u.values, ds.v.values, ds.ang.values, urot, vrot)
        ds['along'] = (('time', 'ypos', 'xpos'), vrot)
        ds['cross'] = (('time', 'ypos', 'xpos'), urot)
    
    #-- creating movie --#
    outp_video = f'{fout}.mp4'
    mov_custom = Movie(ds, custom_quiver, input_check=False)
    mov_custom.save(outp_video, overwrite_existing=True, progress=True)

##############################################################################


if __name__=='__main__':
    main(sys.argv)
