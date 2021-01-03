# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 19:24:01 2020

@author: Leonardo Ramos Emmendorfer

Source code used for the evaluation of the computational time (table 8) in:
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
import functions
import random


import pykrige.kriging_tools as kt
from pykrige.ok import OrdinaryKriging


f_list = [functions.rosen,functions.somb,functions.himm,functions.rast,functions.loggold,functions.f102]
f_names = ['Rosenbrock','Sombrero','Himmelblau','Rastrigin','Log Goldstein-Price','F102']
l_list=[-2.048,0.0,-5.0,-5.12,-2.0,-512]
h_list=[2.048,1.0,5.0,5.12,2.0,512]


limN=1001    #401 for table 7
stepN=200   #100 for table 7

limM=1001   #81 for table 7 
stepM=200  #10 for table 7

NEVAL=6
nrep=30  #300 for table 7
  
minfi=3 #first function   
maxfi=3  #last function
fi=minfi

print(f_names[fi])

print("Columns: N&M& SBS-time & SBS-RMSE & RBF-time&RBF-RMSE & OKG-time& OKG-RMSE  & IDW-time &IDW-RMSE & IDWR-time&IDWR-RMSE:")

for nv in range(stepN,limN,stepN): # nv=number of input points 

    for outt in range(stepM,limM,stepM):   #outt=number of output points
        print(nv,end="&")
        print(outt,end="&")

        time_t=np.zeros((NEVAL,nrep),dtype=float)
        p_error=np.zeros((NEVAL,nrep),dtype=float)

        for rep in range(nrep):
            # choose the function for this replication
        
            f=f_list[fi]
        
            dif=h_list[fi]-l_list[fi]
            pcenter1=l_list[fi]
            pcenter2=l_list[fi]
            # random input
            z=np.zeros(nv,dtype=float)
        
            x=np.random.uniform(pcenter1, pcenter1+dif, nv)
            y=np.random.uniform(pcenter2, pcenter2+dif, nv)
            for i in range(nv):
                z[i]=f(x[i],y[i])      
            fi=fi+1  # next function
            if fi>maxfi:
                fi=minfi

            # a random output point
            xout=np.random.uniform(pcenter1, pcenter1+dif, 1)
            yout=np.random.uniform(pcenter1, pcenter1+dif, 1)
            zout=f(xout,yout)
    
            pretime=time.time()
            for _ in range(outt):  
                zidwv=idw(x,y,z,xout,yout)[0]
     
            time_t[4,rep]=time.time() - pretime
            p_error[4,rep]=(zidwv - zout) ** 2
            
            pretime= time.time() 
            for _ in range(outt):   
                zidwrv=idwr(x,y,z,xout,yout)[0]
            
            time_t[5,rep]=time.time() - pretime        
            p_error[5,rep]=(zidwrv - zout) ** 2

            #Kriging
            pretime= time.time() 
            OKmodel = OrdinaryKriging(x,y,z,variogram_model="gaussian",verbose=False,enable_plotting=False)
            for _ in range(outt):  
                zkrig_pre, ss = OKmodel.execute("grid", xout, xout)
                zkrig=zkrig_pre.data[0]
            time_t[3,rep]=time.time() - pretime
            p_error[3,rep]=(zkrig - zout) ** 2
        
            # RBF
            pretime= time.time() 
            rbfi = Rbf(x, y, z)  # radial basis function interpolator instance
            for _ in range(outt):                               
                zrbf = (rbfi(xout, yout))[0]
            time_t[2,rep]=time.time() - pretime              
            p_error[2,rep]=(zrbf - zout) ** 2         
            #SBS    
            try:
                pretime= time.time() 
                bvs=SmoothBivariateSpline(x,y,z)
                for _ in range(outt):
                    zbvs=bvs.ev(xout,yout)[0]
                time_t[1,rep]=time.time() - pretime                
                p_error[1,rep]=(zbvs - zout) ** 2
            except:
                time_t[1,rep]=-1
                p_error[1,rep]=math.inf
                
        for i in range(NEVAL): 
            if i>0:
                tarray=time_t[i,:]*1000
                    
                time_res1=tarray.mean()
                dp_time1=math.sqrt(np.var(tarray))
                
                print("%10.2f"% time_res1,end="$\pm$")
                print("%10.1f"% dp_time1,end="&")
                print("%0.3f"% math.sqrt(p_error[i,:].mean()),end="&")

        print()

    np.savetxt("time_"+str(nv)+".csv", np.transpose(time_t)*1000, delimiter=",") 
    np.savetxt("perror_"+str(nv)+".csv", np.transpose(p_error), delimiter=",") 


