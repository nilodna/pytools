import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cmocean as cmo
import gsw

def TSdiagram(df, salt='sal00', temp='t090C', depth=None, figsize=(10,10)):
    """Plot a TS diagram with parametric isopycnals based on the limits given in the df

    Parameters
    ----------
    df : pandas.DataFrame
        dataframe containing the T/S pair to plot
    salt : str, optional
        salt variable column name, by default 'sal00'
    temp : str, optional
        temp variable column name, by default 't090C'
    """
    
    # create parametric isopycnals
    T = np.linspace(df[temp].min() - 1, 
                    df[temp].max() + 1,
                    100)
    S = np.linspace(df[salt].min() - 0.1,
                    df[salt].max() + 0.1,
                    100)
    
    T,S = np.meshgrid(T,S)
    
    # compute density
    rho = gsw.sigma0(S,T)
    
    ##-- dataviz --##
    fig,ax = plt.subplots(figsize=figsize)
    
    # plot isopycnals as background
    cr_bground = ax.contour(S, T, rho, colors='grey', 
                            zorder=1, 
                            linewidths=(0.5), 
                            linestyles=('dashed'))
    
    # identification of some isopycnals
    cl_bground = ax.clabel(cr_bground, fontsize=10, inline=True, fmt="%0.1f")
    
    # adding TS pairs
    if depth:
        ts_cr = ax.scatter(df[salt], df[temp], s=10, alpha=1, c=df[depth], cmap=cmo.cm.deep, 
                           edgecolors=('k'), linewidths=(.05))
        cbar = plt.colorbar(ts_cr)
    else:
        ts_cr = ax.scatter(df[salt], df[temp], s=10, alpha=1, color='k')
    
    ax.set_xlabel('Salinity [psu]', fontsize=14)
    ax.set_ylabel(r'Temperature [$^o$C]', fontsize=14)
    
    if depth:
        return fig,ax,cbar
    else:
        return fig,ax

############################################################################################################################

