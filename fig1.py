# -*- coding: utf-8 -*-
"""
Created on Thu Dec 17 14:49:45 2020

@author: Leonardo Ramos Emmendorfer

This is the source code used to produce Fig.1 shown in:
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
import time

import pykrige.kriging_tools as kt
from pykrige.ok import OrdinaryKriging


def func(x1,x2):

    return (x1**3-x1**2+x1)


x=np.zeros(101,dtype=float)
y=np.zeros(101,dtype=float)

z=np.zeros(101,dtype=float)
zkrig=np.zeros(101,dtype=float)
zidw=np.zeros(101,dtype=float)
zidwr=np.zeros(101,dtype=float)
zlin=np.zeros(101,dtype=float)
for i in range(101):

    x[i]=8*i/100.+0.0001-9
    z[i]=func(x[i],0)

xsel=np.array([-5,-6,-7,-8,-9],dtype=float)
ysel=np.zeros(len(xsel),dtype=float)
zsel=np.zeros(len(xsel),dtype=float)
for i in range(len(xsel)):
    zsel[i]=func(xsel[i],0)       
zidw=idw(xsel,ysel,zsel,x,y)
zidwr=idwr(xsel,ysel,zsel,x,y)

OKmodel = OrdinaryKriging(xsel,ysel,zsel,variogram_model="gaussian",verbose=False,enable_plotting=False)

for i in range(len(x)):
    zkrig_pre, ss = OKmodel.execute("grid", [x[i]], [0])
    zkrig[i]=zkrig_pre.data[0][0]


rbfi = Rbf(xsel, ysel, zsel)  # radial basis function interpolator instance                   
zrbf = rbfi(x, y)
    


plt.plot(xsel, zsel, 'ko',x, z, 'k--',x,zidw,'b-',x,zidwr,'r-',x,zrbf,'m-',x,zkrig,'g-')
np.savetxt("x.csv", x, delimiter=",") 
np.savetxt("z.csv", z, delimiter=",") 
np.savetxt("xsel.csv", xsel, delimiter=",") 
np.savetxt("zsel.csv", zsel, delimiter=",") 
np.savetxt("zidw.csv", zidw, delimiter=",") 
np.savetxt("zidwr.csv", zidwr, delimiter=",") 
np.savetxt("zkrig.csv", zkrig, delimiter=",") 
np.savetxt("zrbf.csv", zrbf, delimiter=",") 








 


 

