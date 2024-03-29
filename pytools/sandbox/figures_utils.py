# importing packages and setting figure configurations
from __future__ import unicode_literals
import warnings
warnings.filterwarnings('ignore')

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import cartopy.io.shapereader as shpreader
from matplotlib.patheffects import Stroke
import matplotlib.patches as mpatches
import shapely.geometry as sgeom
from cartopy.io import shapereader

from cartopy.feature import NaturalEarthFeature, LAND, COASTLINE

import os
import numpy as np 
import xarray as xr

def set_figure():
# configuring matplotlib
    from matplotlib import rcParams
    rcParams['figure.figsize']    = [14/2.54,12/2.54] # width for 1.5column large figure
    rcParams['font.size']         = 8
    rcParams['axes.linewidth']    = 0.1
    rcParams['xtick.major.width'] = 0.4
    rcParams['ytick.major.width'] = 0.4

# create cartopy object of a map
def make_map(extent=[-50,-39,-30,-22],ax=None, facecolor=None):
    """Make map using the box send in extent variable, with a minimap on the right
    lower corner.

    Parameters
    ----------
    extent : list
        Coordinates of each corner. [lower left long, upper right long, lower
        left latitude, upper right latitude].

    Usage
    -----
    >>> make_map(extent=[-60,-50,-30,-40])

        HOW TO USE

        #limites
        upper_lat = -22.
        upper_lon = -39.
        lower_lat = -30.
        lower_lon = -49.

        # creating plot
        fig = plt.figure(figsize=(15/2.54,15/2.54))
        ax = fig.add_subplot(1, 2, 1, projection=ccrs.PlateCarree())

        extent = [lower_lon, upper_lon, lower_lat, upper_lat]
        ax = make_map(ax,extent=extent)

        ax = fig.add_subplot(1, 2, 2, projection=ccrs.PlateCarree())
        ax = make_map(ax,extent=extent)

    """

    if ax == None:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

        ax.set_extent(extent, crs=ccrs.PlateCarree())

        
    if facecolor == None:
        facecolor = cfeature.COLORS['land']
        
    # Create a feature for regions at 1:10m from Natural Earth
    states_provinces = cfeature.NaturalEarthFeature(
        category='physical',name='coastline',scale='10m',
        facecolor=facecolor)

    ### ADD FEATURES
    # adding coastline from NaturalEarth dataset
    ax.add_feature(states_provinces, edgecolor='black',linewidth=.1)


    ### CONFIGURING MERIDIANS AND PARALLELS
    gl = ax.gridlines(crs=ccrs.PlateCarree(),draw_labels=True,linewidth=.4,
                        color='gray', alpha=0.5, linestyle='--',
                        xlocs=[extent[0],-49,-46,-43,-40,extent[1]],
                        ylocs=[extent[3],-23,-26,-29,extent[2]])

    gl.xlabels_top = False
    gl.ylabels_right = False
    # gl.xlocator = mticker.FixedLocator([-180, -45, 0, 45, 180])
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.ylabel_style = {'color': 'black', 'fontsize': '8'}
    gl.xlabel_style = {'color': 'black', 'fontsize': '8'}

    return ax

def insert_miniglobe(ax,fig,loc,extent=[-50,-41,-30,-22],center=[-45,-15]):
    
    if loc == 'upper left':
        location = [0.13, 0.642, 0.2, 0.15]
    elif loc == 'lower right':
        location = [0.7, 0.18, 0.2, 0.2]

    sub_ax = fig.add_axes(location,projection=ccrs.Orthographic(center[0],center[1]))

    # Make a nice border around the inset axes.
    effect = Stroke(linewidth=.1, foreground='black', alpha=0.5)
    sub_ax.outline_patch.set_path_effects([effect])

    # Add the land, coastlines and the extent of the Brazil.
    # draw coastlines and borders
    sub_ax.add_feature(cfeature.COASTLINE, lw=0.0001)
    sub_ax.add_feature(cfeature.BORDERS, lw=0.00001)

    # draw meridians and parallels
    gl = ax.gridlines(color='k', linestyle=(0, (1, 1)),
                      xlocs=range(0, 390, 30),
                      ylocs=[-80, -60, -30, 0, 30, 60, 80])

    sub_ax.add_feature(cfeature.LAND,linewidth=0.0001)
    sub_ax.coastlines(linewidth=0.00001)
    sub_ax.stock_img()

    extent_box = sgeom.box(extent[0], extent[2], extent[1], extent[3])
    sub_ax.add_geometries([extent_box], ccrs.PlateCarree(), facecolor='bisque',
                          edgecolor='red', linewidth=1,alpha=.8)

    return sub_ax

def mycolormap2():
    import cmocean as cm
    import matplotlib.colors as mcolors
    
    shades_lightblue = plt.cm.Blues(np.linspace(0,0.7, num=64))
    shades_darkblue  = cm.cm.deep(np.linspace(0.7, 1, num=36))
    colors = np.vstack((shades_lightblue, ))
    
    return mcolors.LinearSegmentedColormap.from_list('my_colormap', colors)

def insert_places(ax):
    # places to show on the map

    # set specific properties for each type of marker
    bbox_state_txt = dict(fontsize=10,ha='center',va='bottom',
                        fontweight='bold',
                        color='darkgray')

    bbox_others_txt   = dict(fontsize=8,
                            ha='center',
                            va='bottom')

    bbox_others_txt2 = dict(fontsize=8,
                            ha='left',
                            va='top')

    others_marker     = dict(marker='o',
                            s=5)

    # setting name, coordinates, and a few styles
    states = {
        'São Paulo':       
            ['SP',(-23.541518, -47.396635),bbox_state_txt],
        'Rio de Janeiro':       
            ['RJ',(-22.665077, -43.296820),bbox_state_txt],
    }

    cities = {
        'Santos':           
            ['Santos',            (-23.967573, -46.326515),bbox_others_txt,others_marker],    
        'Cabo Frio':        
            ['Cabo Frio',         (-22.888983, -42.026578),bbox_others_txt,others_marker],
        'Ubatuba':        
            ['Ubatuba',           (-23.437697, -45.085100),bbox_others_txt,others_marker],
        u'Cananéia':        
            [u'Cananéia',         (-24.838208, -47.716180),bbox_others_txt,others_marker],
        u'Santa Marta Cape':
            [u'Santa Marta Cape', (-28.115909, -48.657727),bbox_others_txt,others_marker],
        u'São Tomé Cape':
            [u'São Tomé Cape',    (-21.982736, -40.982003),bbox_others_txt2,others_marker],
    }
        
    # inserting material and methods elements [numerical grid, transects, observation locations]

    ### ---- labels for states and cities --- ###
    for place in states:
        data   = states[place]
        name   = data[0]
        coord  = data[1]
        kwargs = data[2]
        
        ax.text(coord[1],coord[0],name,transform=ccrs.PlateCarree(),zorder=5,**kwargs)

    for place in cities:
        data   = cities[place]
        name   = data[0]
        coord  = data[1]
        kwargs1= data[2]
        kwargs2= data[3]
        
        ax.text(coord[1],coord[0],name,transform=ccrs.PlateCarree(),zorder=5,**kwargs1)
        ax.scatter(coord[1],coord[0],c='k',transform=ccrs.PlateCarree(),zorder=6,**kwargs2)

def crop_image(fname):
    # function to remove white spaces from pdf and png figures
    extension = fname.split('.')[-1]
    if extension == 'png':
        os.system('convert -trim %s %s'%(fname,fname))
    elif extension == 'pdf':
        os.system('pdfcrop %s %s'%(fname,fname))
