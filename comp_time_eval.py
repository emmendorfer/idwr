# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 19:24:01 2020

@author: Leonardo Ramos Emmendorfer

Source code used for the evaluation of the computational time (table 7) in:
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

#an arbitrary test function 
def testf(x1,x2):        
    return(math.sin(x1+x2))

limN=31
stepN=10

limM=11
stepM=1

NEVAL=6
dif=10;
pcenter1=0
pcenter2=0

nrep=300

for nv in range(stepN,limN,stepN): # nv=number of input points 

    z=np.zeros(nv,dtype=float)

    x=np.random.uniform(pcenter1, pcenter1+dif, nv)
    y=np.random.uniform(pcenter2, pcenter2+dif, nv)
    for i in range(nv):
        z[i]=testf(x[i],y[i])
    for outt in range(stepM,limM,stepM):   #outt=number of output points
        print(nv,end="&")
        print(outt,end="&")
      
        # an arbitraty input point
        xout=[0.5]
        yout=[0.5]
    
        time_t=np.zeros((NEVAL,nrep),dtype=float)
        for rep in range(nrep):

    
            pretime=time.time()
            for _ in range(outt):  
                zidwv=idw(x,y,z,xout,yout)
    
    
            time_t[4,rep]=time.time() - pretime
            pretime= time.time() 
            for _ in range(outt):   
                zidwrv=idwr(x,y,z,xout,yout)
             
            time_t[5,rep]=time.time() - pretime

            #Kriging
            pretime= time.time() 
            OKmodel = OrdinaryKriging(x,y,z,variogram_model="gaussian",verbose=False,enable_plotting=False)
            for _ in range(outt):  
                zkrig_pre, ss = OKmodel.execute("grid", xout, xout)
                zkrig=zkrig_pre.data
            time_t[3,rep]=time.time() - pretime
        
            # RBF
            pretime= time.time() 
            rbfi = Rbf(x, y, z)  # radial basis function interpolator instance
            for _ in range(outt):                               
                zrbf = (rbfi(xout, yout))
            time_t[2,rep]=time.time() - pretime              
            
            #SBS    
            try:
                pretime= time.time() 
                bvs=SmoothBivariateSpline(x,y,z)
                for _ in range(outt):
                    zbvs=bvs.ev(xout,yout)[0]  
                time_t[1,rep]=time.time() - pretime                  
         
            except:
                time_t[1,rep]=-1

        print(" SBS & RBF & OKG & IDW & IDWR:")
        for i in range(NEVAL): 
            if i>0:
                tarray=time_t[i,:]*1000
    
                time_res1=tarray.mean()
                dp_time1=math.sqrt(np.var(tarray))
                print("%10.2f"% time_res1,end="$\pm$")
                print("%10.2f"% dp_time1,end="&")
        print()

    np.savetxt("time_"+str(nv)+".csv", np.transpose(time_t)*1000, delimiter=",") 

#SBS,RBF,OKG,IDW,IDWR
