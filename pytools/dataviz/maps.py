from __future__ import unicode_literals
import numpy as np
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
from matplotlib import patheffects
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import cartopy.io.shapereader as shpreader
from matplotlib.patheffects import Stroke
import shapely.geometry as sgeom
from cartopy.io import shapereader

def make_map(extent=[-50,-41,-30,-22], 
             ax=None, 
             add_gridlines=False,
             add_features=False,
             ):
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

    """

    if ax == None:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

    ax.set_extent(extent, crs=ccrs.PlateCarree())

    if add_gridlines:
        ### CONFIGURING MERIDIANS AND PARALLELS
        gl = ax.gridlines(crs=ccrs.PlateCarree(),draw_labels=True,linewidth=.4,
                            color='gray', alpha=0.5, linestyle='--',
                            xlocs=[extent[0],-49,-46,-43,extent[1]],
                            ylocs=[extent[3],-23,-26,-29,extent[2]])

        gl.xlabels_top = False
        gl.ylabels_right = False
        # gl.xlocator = mticker.FixedLocator([-180, -45, 0, 45, 180])
        gl.xformatter = LONGITUDE_FORMATTER
        gl.yformatter = LATITUDE_FORMATTER
        gl.ylabel_style = {'color': 'black', 'fontsize': '8','weight': 'bold'}
        gl.xlabel_style = {'color': 'black', 'fontsize': '8','weight': 'bold'}

    # add high resolution coastline from openstreetmap
    if add_features:
        ax = add_openstreetmap_shapefile(ax)

    return ax

############################################################################################################################

def add_openstreetmap_shapefile(ax, 
                                landcolor=cfeature.COLORS['land'],
                                coastline_color='white',
                                path='/home/danilo/Research/data/shapefiles/OSM'):
    """Function to add a coastline and land (colored by the given colors) from the Open Street Map dataset. 
    Both information are of high resolution, describing several features nearshore, like small islands, beachs, and so on.

        Parameters
    ----------
    ax : cartopy.mpl.geoaxes.GeoAxesSubplot
        Axes with ccrs.PlateCarree() projection already set for a given domain.
    path : str, optional
        full path to the openstreetmap dataset, by default '/home/danilo/Research/data/shapefiles/OSM'

    Returns
    -------
    cartopy.mpl.geoaxes.GeoAxesSubplot
    """

    # adding coastline from openstreemap
    line = shapereader.Reader(f'{path}/OSM_BRA_coastline/lines.shp')
    ax.add_geometries(line.geometries(), ccrs.PlateCarree(), facecolor=coastline_color, edgecolor='black')

    # adding land from openstreemap
    land = shapereader.Reader(f'{path}/OSM_BRA_land/land.shp')
    ax.add_geometries(land.geometries(), ccrs.PlateCarree(), facecolor=landcolor, edgecolor='black')

    return ax

############################################################################################################################

def insert_miniglobe(ax, 
                     fig, 
                     extent=[-50,-41,-30,-22], 
                     center=[-45,-15], 
                     location=None,
                     box_color='red',
                     box_alpha=1,
                     box_edge_width=1):
    # upper left: [0.13, 0.642, 0.2, 0.15]
    # lower right: [0.7, 0.18, 0.2, 0.2]
    # 
    if not location:
        location = [0.75, 0.18, 0.2, 0.2]

    sub_ax = fig.add_axes(location, projection=ccrs.Orthographic(center[0],center[1]))

    # Make a nice border around the inset axes.
    effect = Stroke(linewidth=.1, foreground='black', alpha=0.5)
    sub_ax.outline_patch.set_path_effects([effect])

    sub_ax.coastlines(linewidth=0.00000001, edgecolor='k', alpha=.8)
    sub_ax.stock_img()

    extent_box = sgeom.box(extent[0], extent[2], extent[1], extent[3])
    sub_ax.add_geometries([extent_box], ccrs.PlateCarree(), facecolor='bisque',
                          edgecolor=box_color, linewidth=box_edge_width, alpha=box_alpha)

    return sub_ax

############################################################################################################################

def scale_bar(ax, length=None, location=(0.5, 0.05), linewidth=3, Narrow=False):
    """
        ax is the axes to draw the scalebar on.
        length is the length of the scalebar in km.
        location is center of the scalebar in axis coordinates.
        (ie. 0.5 is the middle of the plot)
        linewidth is the thickness of the scalebar.
    """
    #Returns numbers starting with the list
    def scale_number(x):
        if str(x)[0] in ['1', '2', '5']: return int(x)
        else: return scale_number(x - 10 ** ndim)
        
    #Get the limits of the axis in lat long
    llx0, llx1, lly0, lly1 = ax.get_extent(ccrs.PlateCarree())
    #Make tmc horizontally centred on the middle of the map,
    #vertically at scale bar location
    sbllx = (llx1 + llx0) / 2
    sblly = lly0 + (lly1 - lly0) * location[1]
    tmc = ccrs.TransverseMercator(sbllx, sblly)
    #Get the extent of the plotted area in coordinates in metres
    x0, x1, y0, y1 = ax.get_extent(tmc)
    #Turn the specified scalebar location into coordinates in metres
    sbx = x0 + (x1 - x0) * location[0]
    sby = y0 + (y1 - y0) * location[1]-0.05

    #Calculate a scale bar length if none has been given
    #(Theres probably a more pythonic way of rounding the number but this works)
    if not length:
        length = (x1 - x0) / 5000 #in km
        ndim = int(np.floor(np.log10(length))) #number of digits in number
        length = round(length, -ndim) #round to 1sf

        length = scale_number(length)

    #Generate the x coordinate for the ends of the scalebar
    bar_xs = [sbx - length * 500, sbx + length * 500]
    #Plot the scalebar
    ax.plot(bar_xs, [sby, sby], transform=tmc, color='k', linewidth=linewidth)
    # #Plot the scalebar label
    ax.text(sbx, sby, str(length) + ' km', transform=tmc,
            horizontalalignment='center', verticalalignment='bottom')

    # buffer for text
    buffer = [patheffects.withStroke(linewidth=3, foreground="w")]

    left = llx0+(llx1-llx0)*0.05
    # Plot the N arrow
    if Narrow:
        sby_arrow = location[1]
        t1 = ax.text(location[0], location[1]+0.03, u'\u25B2\nN', transform=ax.transAxes,
            horizontalalignment='center', verticalalignment='bottom',
            path_effects=buffer, zorder=2)

############################################################################################################################

def create_coastline(ax, scale='10m'):
    """Add default coastline from NaturalEarth dataset using cartopy methods.

    Parameters
    ----------
    ax : cartopy.mpl.geoaxes.GeoAxesSubplot
        [description]
    scale : str, optional
        coastline resolution, by default '10m'

    Returns
    -------
    cartopy.mpl.geoaxes.GeoAxesSubplot
        [description]
    """
    # plot coastline using NaturalEarth Dataset, with a 10-m scale
    coastline_10m = cfeature.NaturalEarthFeature(
        category='physical',name='coastline',scale=scale,
        facecolor=cfeature.COLORS['land'])
    
    ax.add_feature(coastline_10m, edgecolor='black',linewidth=.1)   # coastline

    return ax

############################################################################################################################