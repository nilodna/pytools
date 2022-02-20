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

from pytools.sandbox.sandbox import crop_image

# return an extent dict structure to be used in the ax.set_extent() method
def get_extent():
    # set area limits
    upper_lat = -21.
    upper_lon = -39.
    lower_lat = -30.
    lower_lon = -50.

    extent = [lower_lon, upper_lon, lower_lat, upper_lat]
    
    return extent

#####################################################################################################################

# create South Brazil Bight map, returning an ax object
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

#####################################################################################################################

# function to insert important places on the South Brazil Bight. Check Silva and Dottori (2021), Fig 2 for an example.
def insert_places(ax, language='eng'):
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

    dictionary = {
        'Santos': {
            'eng': 'Santos',
            'ptbr': 'Santos',
            'location': (-23.967573, -46.326515),
            'bbox1': bbox_others_txt,
            'bbox2': others_marker
        },
        'Cabo Frio': {
            'eng': 'Cabo Frio',
            'ptbr': 'Cabo Frio',
            'location': (-22.888983, -42.026578),
            'bbox1': bbox_others_txt,
            'bbox2': others_marker
        },
        'Ubatuba': {
            'eng': 'Ubatuba',
            'ptbr': 'Ubatuba',
            'location': (-23.437697, -45.085100),
            'bbox1': bbox_others_txt,
            'bbox2': others_marker
        },
        'Cananéia': {
            'eng': 'Cananéia',
            'ptbr': 'Cananéia',
            'location': (-24.838208, -47.716180),
            'bbox1': bbox_others_txt,
            'bbox2': others_marker
        },
        'Santa Marta': {
            'eng': 'Santa Marta Cape',
            'ptbr': 'Cabo de\nSanta Marta',
            'location': (-28.115909, -48.657727),
            'bbox1': bbox_others_txt,
            'bbox2': others_marker
        },
        u'São Tomé': {
            'eng': u'São Tomé Cape',
            'ptbr': u'Cabo de\nSão Tomé',
            'location': (-21.982736, -40.982003),
            'bbox1': bbox_others_txt2,
            'bbox2': others_marker
        },
    }

    # setting name, coordinates, and a few styles
    states = {
        'São Paulo':       
            ['SP',(-23.541518, -47.396635),bbox_state_txt],
        'Rio de Janeiro':       
            ['RJ',(-22.665077, -43.296820),bbox_state_txt],
    }

    # cities = {
    #     'Santos':           
    #         ['Santos',            (-23.967573, -46.326515),bbox_others_txt,others_marker],    
    #     'Cabo Frio':        
    #         ['Cabo Frio',         (-22.888983, -42.026578),bbox_others_txt,others_marker],
    #     'Ubatuba':        
    #         ['Ubatuba',           (-23.437697, -45.085100),bbox_others_txt,others_marker],
    #     u'Cananéia':        
    #         [u'Cananéia',         (-24.838208, -47.716180),bbox_others_txt,others_marker],
    #     u'Santa Marta Cape':
    #         [u'Santa Marta Cape', (-28.115909, -48.657727),bbox_others_txt,others_marker],
    #     u'São Tomé Cape':
    #         [u'São Tomé Cape',    (-21.982736, -40.982003),bbox_others_txt2,others_marker],
    # }
        
    # inserting material and methods elements [numerical grid, transects, observation locations]

    ### ---- labels for states and cities --- ###
    for place in states:
        data   = states[place]
        name   = data[0]
        coord  = data[1]
        kwargs = data[2]
        
        ax.text(coord[1],coord[0],name,transform=ccrs.PlateCarree(),zorder=5,**kwargs)

    # for place in cities:
    #     data   = cities[place]
    #     name   = data[0]
    #     coord  = data[1]
    #     kwargs1= data[2]
    #     kwargs2= data[3]
    for place,data in dictionary.items():
        name = data[language]
        coord = data['location']
        kwargs1 = data['bbox1']
        kwargs2 = data['bbox2']
        
        ax.text(coord[1],coord[0],name,transform=ccrs.PlateCarree(),zorder=5,**kwargs1)
        ax.scatter(coord[1],coord[0],c='k',transform=ccrs.PlateCarree(),zorder=6,**kwargs2)

#####################################################################################################################


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

def sbb_study_area(fname, language='eng'):
    language = 'ptbr'
    
    textual_elements = {
        'bathymetry': {'ptbr': 'Batimetria [m]', 'eng': 'Bathymetry [m]'},
        'isobath': {'ptbr': u'Isóbatas [m]', 'eng': 'Isobaths [m]'}
    }
    
    # set area limits
    upper_lat = -21.
    upper_lon = -39.
    lower_lat = -30.
    lower_lon = -50.

    extent = [lower_lon, upper_lon, lower_lat, upper_lat]

    # set element colors
    land         = cfeature.COLORS['land']
    ocean        = 'lightblue'
    state_limits = 'gray'

    ### --- Creating figure's structure --- ###
    fig = plt.figure(figsize=(17/2.54,17/2.54))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

    # setting region
    ax.set_extent(extent, crs=ccrs.PlateCarree())

    # inserting coastline
    coastline = cfeature.NaturalEarthFeature(category='physical',
                                            name='coastline',
                                            scale='10m',
                                            facecolor=land)

    ax.add_feature(coastline, edgecolor=land, linewidth=.01)

    # inserting states limits
    limits    = cfeature.NaturalEarthFeature(category='cultural',
                                            name='admin_1_states_provinces_shp',
                                            scale='10m',
                                            facecolor='none')

    ax.add_feature(limits, edgecolor=state_limits,linewidth=.1)
    
    # add gridline [meridians and parallells]
    gl = ax.gridlines(crs=ccrs.PlateCarree(),draw_labels=True,linewidth=.4,
                        color='gray', alpha=0.5, linestyle='--',
                        xlocs=[extent[0],-49,-46,-43,-40,extent[1]],
                        ylocs=[extent[3],-23,-26,-29,extent[2]])

    gl.xlabels_top = False
    gl.ylabels_right = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.ylabel_style = {'color': 'black', 'fontsize': '8'}
    gl.xlabel_style = {'color': 'black', 'fontsize': '8'}

    ### --- adding places --- ###
    insert_places(ax)

    ### --- bathymetry -- ##
    grid_lon = np.load(os.path.join('/media/Danilo/Research/MSc/summer2014circulation/data','model_outputs','lon.npy'))
    grid_lat = np.load(os.path.join('/media/Danilo/Research/MSc/summer2014circulation/data','model_outputs','lat.npy'))

    # bathymetry
    dep = np.load(('/media/Danilo/Research/MSc/summer2014circulation/data/model_outputs/dep.npy'))

    grid_kwargs = dict(linewidth=.1,alpha=.05,transform=ccrs.PlateCarree())
    ax.plot(grid_lon[2:,:],grid_lat[2:,:],'k',**grid_kwargs);
    ax.plot(grid_lon[2:,:].T,grid_lat[2:,:].T,'k',**grid_kwargs);
        
    ## -- contourf for bathymetry -- ##
    cf = ax.contourf(grid_lon[2:,:],
                    grid_lat[2:,:],
                    dep[2:,:],
                    np.arange(0,4001,50),
                    cmap=mycolormap2())

    ## -- contour as isobaths -- ##
    cr = ax.contour(grid_lon[2:,:],
                    grid_lat[2:,:],
                    dep[2:,:],
                    levels=[40,80,200],
                    colors=('gray'),
                    linestyles=('--','-',':'),
                    linewidths=(.5,.5,.5))

    ## -- rectangle for contour legend -- ## (-43.0,-22.2),
    ax.add_patch(mpatches.Rectangle(xy=(-43.5,-22.),
                                    width=4.0,height=.8,
                                    facecolor='white',alpha=.6,zorder=5,
                                    transform=ccrs.PlateCarree()))

    ## -- isobath legend -- ##
    labels = ['40','80','200']
    for i in range(len(labels)):
        cr.collections[i].set_label(labels[i])
        
    h1 = cr.legend_elements()
    handles = []
    for handle in h1:
        handles.append(handle)
        
    handles = [handles[0][0],handles[0][1],handles[0][2]]
    plt.legend(handles=handles,labels=labels,ncol=3,loc='upper right',
            bbox_to_anchor=(.95,.95),
            framealpha=.7,frameon=False,fontsize=8,
            columnspacing=1.)

    # legend title
    ax.text(-42.3,-21.5, textual_elements['isobath'][language],transform=ccrs.PlateCarree(),fontsize=8,zorder=6)

    ##############################################################################################
    ## -- inserting colorbar for bathymetry only after contour to avoid compatibility issues -- ##
    ##############################################################################################

    ## -- rectangle for contourf colorbar -- ##
    ax.add_patch(mpatches.Rectangle(xy=(-44.87,-29.9),
                                    width=5.5, height=1.3, edgecolor='lightgray',
                                    facecolor='white', alpha=.6, zorder=5,
                                    transform=ccrs.PlateCarree()))

    ## -- colorbar -- ##
    #caxes = plt.axes([0.15, 0.23, 0.34, 0.01]) # colorbar fixed at lower left
    caxes = plt.axes([0.5, 0.23, 0.34, 0.01]) # colorbar fixed at lower right
    cbar  = plt.colorbar(cf,cax=caxes,
                        orientation='horizontal',
                        ticks=[0,800,1600,2400,3200,4000]
                        )

    ## -- convert depth values to negative -- ##
    new_cbar_values = [int(lab.get_text())*(-1) for lab in cbar.ax.get_xticklabels()]
    cbar.ax.set_xticklabels(new_cbar_values)  # horizontal colorbar
    cbar.ax.axes.tick_params(axis='both',which='both',labelsize=8)

    cbar.ax.set_title(textual_elements['bathymetry'][language],fontsize=8)

    ### ---- inserting miniglobe on the upper left corner --- ###
    sub_ax = insert_miniglobe(ax, fig, loc='upper left')

    # saving high resolution figure
    plt.savefig(fname, dpi=300)
    crop_image(fname)
