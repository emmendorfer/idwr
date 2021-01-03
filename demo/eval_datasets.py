# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 19:24:01 2020

@author: Leonardo Ramos Emmendorfer


Source code used for the evaluation of IDWR algorithm on real-world datasets (table 6) in:
 L. R. Emmendorfer, G. P. Dimuro, A point interpolation algorithm resulting
 from weighted linear regression (to appear)
"""

import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt
import pandas as pd
from idwr import idw
from idwr import idwr
from scipy.interpolate import Rbf
from scipy.interpolate import LinearNDInterpolator
from scipy.interpolate import SmoothBivariateSpline
import math
import pykrige.kriging_tools as kt
from pykrige.ok import OrdinaryKriging


def crossval(filename):
    arqle=pd.read_csv(filename,names=None,header=None)
    vec=arqle.to_numpy();

    x=vec[:,0]
    y=vec[:,1]
    z=vec[:,2]

    nv=len(x)
    rmse_idw=np.zeros(nv)
    rmse_idwr=np.zeros(nv)
    rmse_rbf=np.zeros(nv)
    rmse_bvs=np.zeros(nv)
    rmse_krig=np.zeros(nv)
    for pos in range(nv):
        xsel=x[np.arange(nv)!=pos]
        ysel=y[np.arange(nv)!=pos]
        zsel=z[np.arange(nv)!=pos]
        zidw=(idw(xsel,ysel,zsel,[x[pos]],[y[pos]]))[0]
        zidwr=(idwr(xsel,ysel,zsel,[x[pos]],[y[pos]]))[0]
        rmse_idw[pos]=(zidw - z[pos]) ** 2
        rmse_idwr[pos]=(zidwr - z[pos]) ** 2

        OKmodel = OrdinaryKriging(xsel,ysel,zsel,variogram_model="gaussian",verbose=False,enable_plotting=False)

        zkrig_pre, ss = OKmodel.execute("grid", [x[pos]*1.], [y[pos]*1.])
        zkrig=zkrig_pre.data[0][0]
        rmse_krig[pos]=(zkrig - z[pos]) ** 2
        try:
            rbfi = Rbf(xsel, ysel, zsel)  # radial basis function interpolator instance
            zrbf = (rbfi([x[pos]], [y[pos]]))[0]
            rmse_rbf[pos]=(zrbf - z[pos]) **2
        except:
            e=0#print("Error RBF")


        try:
            bvs=SmoothBivariateSpline(xsel,ysel,zsel)
            zbvs=bvs.ev([x[pos]],[y[pos]])[0]            
            rmse_bvs[pos]=(zbvs - z[pos]) ** 2            
        except:
            e=0;#print("Error BVS")
    error_idw=np.sqrt(rmse_idw.mean())
    error_idwr=np.sqrt(rmse_idwr.mean())
    error_rbf=np.sqrt(rmse_rbf.mean())
    error_bvs=np.sqrt(rmse_bvs.mean())
    error_krig=np.sqrt(rmse_krig.mean())
    
    return error_bvs,error_rbf,error_krig,error_idw,error_idwr

    
# Real-world datasets

error_rw=np.zeros((4,5),dtype=float)
error_rw[0,0],error_rw[0,1],error_rw[0,2],error_rw[0,3],error_rw[0,4]=crossval("datasets/calabria.csv")
error_rw[1,0],error_rw[1,1],error_rw[1,2],error_rw[1,3],error_rw[1,4]=crossval("datasets/cretaceous.csv")  
error_rw[2,0],error_rw[2,1],error_rw[2,2],error_rw[2,3],error_rw[2,4]=crossval("datasets/texas.csv")
error_rw[3,0],error_rw[3,1],error_rw[3,2],error_rw[3,3],error_rw[3,4]=crossval("datasets/amazon.csv")    

float_formatter = "{:.2f}".format
np.set_printoptions(formatter={'float_kind':float_formatter})


print("Rows:")
print("Calabria")
print("Cretaceous")
print("Texas")
print("Amazon")
print("Columns: SBS, RBF, OK-G, IDW, IDWR")
print("RMSE:")
print(error_rw)

