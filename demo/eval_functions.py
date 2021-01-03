# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 19:24:01 2020

@author: Leonardo Ramos Emmendorfer

Source code used for the evaluation of IDWR algorithm (table 5) in:
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
from functions import somb
from functions import rosen
from functions import rast
from functions import loggold
from functions import himm
from functions import f102

import pykrige.kriging_tools as kt
from pykrige.ok import OrdinaryKriging

#Function evaluations 
print("RMSE:")
print()
nrep=30
NEVAL=7

f_list = [rosen,somb,himm,rast,loggold,f102]
f_names = ['Rosenbrock','Sombrero','Himmelblau','Rastrigin','Log Goldstein-Price','F102']
l_list=[-2.048,0.0,-5.0,-5.12,-2.0,-512]
h_list=[2.048,1.0,5.0,5.12,2.0,512]


for ind in range(4): # nunmber of input points divided by 100

   
    error_f=np.zeros((6,NEVAL),dtype=float)
    dp_f=np.zeros((6,NEVAL),dtype=float)
    nv=(ind+1)*100
    erroridw=np.zeros((nrep,6),dtype=float)
    erroridwr=np.zeros((nrep,6),dtype=float)
    errorbvs=np.zeros((nrep,6),dtype=float)
    errorrbf=np.zeros((nrep,6),dtype=float)
    errorlin=np.zeros((nrep,6),dtype=float)
    errorkrig=np.zeros((nrep,6),dtype=float)
        
    #lin bvs rbf Ok OkMatern idw idwr
    
    fi=0
    print()

    print("Size ",nv,end=": ")
    for f in f_list:   #for all functions
        print(f_names[fi],end=" ")
        for rep in range(nrep):
             print("*",end="") 
             x=np.zeros(nv,dtype=float)
             y=np.zeros(nv,dtype=float)
             z=np.zeros(nv,dtype=float)

             dif=h_list[fi]-l_list[fi];

             pcenter1=l_list[fi]
             pcenter2=l_list[fi]

             x=np.random.uniform(pcenter1, pcenter1+dif, nv)
             y=np.random.uniform(pcenter2, pcenter2+dif, nv)
             for i in range(nv):
                 z[i]=f(x[i],y[i])


             p_erroridw=np.zeros(nv,dtype=float)
             p_erroridwr=np.zeros(nv,dtype=float)
             p_errorbvs=np.zeros(nv,dtype=float)
             p_errorrbf=np.zeros(nv,dtype=float)

             p_errorkrig=np.zeros(nv,dtype=float)
             for pos in range(nv):
                 xsel=x[np.arange(nv)!=pos]
                 ysel=y[np.arange(nv)!=pos]
                 zsel=z[np.arange(nv)!=pos]
       
                 zidwv=idw(xsel,ysel,zsel,[x[pos]],[y[pos]])
                 zidw=zidwv[0]
                 zidwrv=idwr(xsel,ysel,zsel,[x[pos]],[y[pos]])
                 zidwr=zidwrv[0]

                 p_erroridw[pos]=(zidw - z[pos]) ** 2
                 p_erroridwr[pos]=(zidwr - z[pos]) ** 2
                 
                 #Kriging
                 pretime= time.time() 
                 OKmodel = OrdinaryKriging(xsel,ysel,zsel,variogram_model="gaussian",verbose=False,enable_plotting=False)
                 
                 pretime= time.time() 
                 zkrig_pre, ss = OKmodel.execute("grid", [x[pos]], [y[pos]])
                 zkrig=zkrig_pre.data[0][0]
                
                 p_errorkrig[pos]=(zkrig - z[pos]) ** 2
                 try:
                     # RBF
                     pretime= time.time() 
                     rbfi = Rbf(xsel, ysel, zsel)  # radial basis function interpolator instance
                    
                     pretime= time.time()                      
                     zrbf = (rbfi([x[pos]], [y[pos]]))[0]
                     p_errorrbf[pos]=(zrbf - z[pos]) ** 2
                 except:
                     e=0#print("Error RBF")
             
                 try:
                     #SBS    
                     pretime= time.time() 
                     bvs=SmoothBivariateSpline(xsel,ysel,zsel)
                   
                     pretime=time.time()
                     zbvs=bvs.ev([x[pos]],[y[pos]])[0]  
                                     
                     p_errorbvs[pos]=(zbvs - z[pos]) ** 2            
                 except:
                     e=0;#print("Error SBS")
         
 
             erroridw[rep,fi]=np.sqrt(p_erroridw.mean())
             erroridwr[rep,fi]=np.sqrt(p_erroridwr.mean())
             errorrbf[rep,fi]=np.sqrt(p_errorrbf.mean())
             errorbvs[rep,fi]=np.sqrt(p_errorbvs.mean())
             errorkrig[rep,fi]=np.sqrt(p_errorkrig.mean())

        error_f[fi,1]=(errorbvs[:,fi].mean())
        error_f[fi,2]=(errorrbf[:,fi].mean())
        error_f[fi,3]=(errorkrig[:,fi].mean())
        error_f[fi,4]=(erroridw[:,fi].mean())
        error_f[fi,5]=(erroridwr[:,fi].mean())


        fi=fi+1
    np.savetxt("erroridw_"+str(100*(ind+1))+".csv", erroridw, delimiter=",")    
    np.savetxt("erroridwr_"+str(100*(ind+1))+".csv", erroridwr, delimiter=",")  
    np.savetxt("errorrbf_"+str(100*(ind+1))+".csv", errorrbf, delimiter=",")  
    np.savetxt("errorbvs_"+str(100*(ind+1))+".csv", errorbvs, delimiter=",") 
    np.savetxt("errorkrig_"+str(100*(ind+1))+".csv", errorkrig, delimiter=",") 
                     

    float_formatter = "{:.3f}".format
    np.set_printoptions(formatter={'float_kind':float_formatter})
    print()
    print(nv)
    
    print()
    for i in range(len(error_f)):
            for j in range(NEVAL-1):
                if j>0:               
                    print("%10.3f"% error_f[i][j],end=" ")
            print()    

    print("Rows:")
    for i in range(6):
        print(f_names[i])
    print("Columns: SBS, RBF, OK-G, IDW, IDWR")
   

