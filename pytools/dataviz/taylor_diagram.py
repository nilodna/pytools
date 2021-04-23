import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#####################################################################################

def taylor(fig, thetamax=np.pi/2, stdlim=2, figsize=[3.42, 3.42], skill=False):
    """
    Made by Dalton K. Sasaki
    
    usage
    -----
        phase = np.pi/10
        magnt = 1.5
        x = np.arange(0,1,0.001)
        reference = np.sin(x*np.pi*10)
        synt_mesr = magnt*np.sin(x*np.pi*10+np.pi/10)
        fig = plt.figure()
        ax =  taylor(fig, skill=True)
        # convert from cartesian to polar coordinates
        theta_corr = np.arccos(np.corrcoef(reference, synt_mesr)[0,1])
        std0 = np.std(reference)
        std1 = np.std(synt_mesr)
        radius_stdn = std1/std0  # divide by the standard deviation
        ax.scatter(theta_corr, radius_stdn)  # plot
    """
    fig.add_subplot(111, projection='polar')
    ax = plt.gca()
    ax.set_ylim(0, stdlim)
    ax.set_xlim(0, thetamax)
    
    rlocs = np.array([0, 0.2, 0.4, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1])
    tlocs = np.arccos(rlocs)        # Conversion to polar angles
    
    ax.set_xticks(tlocs)
    ax.set_xticklabels(rlocs, minor=False, rotation=45)
    
    ax.text(0.75,0.75, 'correlation',  rotation=-55, fontsize=8, transform=ax.transAxes)
    ax.text(0.25,-0.1, 'standard deviation (norm.)', fontsize=8, transform=ax.transAxes)
    ax.grid(True, linestyle='--', linewidth=0.5)
    
    if skill:
        xm,ym,zm = skill_willmott_space()
        lims = [0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.975, 0.99]
    else:
        xm, ym = np.meshgrid(np.linspace(0,thetamax, 100), np.linspace(0,stdlim,100))
        zm = np.sqrt(1**2+ym**2 - 2*ym*np.cos(xm))
        lims = np.arange(0.2, 10, 0.4)
        
    cr = ax.contour(xm, ym, zm, lims, linewidths=0.5, colors='0.5')
    
    plt.clabel(cr, fontsize=6)
    ax.plot(np.linspace(0, thetamax,100), np.ones(100), c='k', linewidth=0.5)
    
    return ax

#####################################################################################

def skill_willmott(re,m):
    """
    Analise de Skill (Willmott, 1981) em Python
    Esta funcao esta definida como no trabalho de Miranda et al 2012 (BRAZILIAN JOURNAL OF OCEANOGRAPHY, 60(1):11-23, 201)
    CIRCULATION AND SALT INTRUSION IN THE PIACAGUERA CHANNEL, SANTOS (SP)
    Based on the MSE , a quantitative model skill was presented by Willmott (1981)
    The highest value, WS = 1, means perfect agreement between model and observation, while the lowest value,  WS = 0,
    indicates   complete     disagreement. Recently, this was used to evaluate ROMS in the simulation of multiple parameters in the
    Hudson River estuary [ Warner et al., 2005b] and on the southeast New England Shelf [Wilkin ,2006].
    The Willmott skill will be used to  quantify model performance in simulating different parameters from the best model run
    skill parameter (WILLMOTT, 1981)
    Parameters:
    re - real data
    m - model data
    skill - Skill parameter
    funcao traduzida por: Paula Birocchi
    """
    dif   = re - m
    soma  = np.nansum(abs(dif)**2)
    somam = m - np.nanmean(re)
    c     = re - np.nanmean(re)
    d     = np.nansum((abs(somam) + abs(c))**2)
    skill = 1 - (soma/d)
    return skill

#####################################################################################

def skill_willmott_space():
    willmott = np.zeros([100,100])
    corrcoef = np.zeros([100,100])
    std = np.zeros([100,100])
    x = np.arange(0.001,1,0.001)
    for im, m in enumerate(np.linspace(0,2,100)):
        for ip, p in enumerate(np.linspace(0,1,100)):
            y0 = np.sin(x*np.pi*10)  # reference data
            y1 = m*np.sin(x*np.pi*10+np.pi*p)  # 'measurement'
            willmott[im,ip] = skill_willmott(y0,y1)
            corrcoef[im,ip] = np.arccos(np.corrcoef(y0,y1)[0,1])
            std0 = np.std(y0)
            std1 = np.std(y1)
            std[im,ip] = std1/std0
    return corrcoef, std, willmott


#####################################################################################

if __name__ == '__main__':
    # -- synthetic data -- #
    phase = np.pi/10  # phase difference between reference and synt_mes
    magnt = 1.5  # magnitude factor
    x = np.arange(0,1,0.001)
    reference = np.sin(x*np.pi*10)  # 'reference'
    synt_mesr = magnt*np.sin(x*np.pi*10+np.pi/10)  # 'measurement'
    # -- plot time series -- #
    plt.figure()
    plt.scatter(x,reference)
    plt.scatter(x,synt_mesr)
    # -- taylor diagram -- #
    fig = plt.figure()
    ax =  taylor(fig, skill=True)
    # convert from cartesian to polar coordinates
    theta_corr = np.arccos(np.corrcoef(reference, synt_mesr)[0,1])
    std0 = np.std(reference)
    std1 = np.std(synt_mesr)
    radius_stdn = std1/std0  # divide by the standard deviation
    ax.scatter(theta_corr, radius_stdn)  # plot
    # if the textlabels are in a weird position, you can adjust them
    # in the 'taylor' function (look for ax.text)
