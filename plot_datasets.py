# -*- coding: utf-8 -*-
"""
Created on Thu Dec 17 14:14:04 2020

@author: Leonardo Ramos Emmendorfer
"""
import pandas as pd
from scipy.interpolate import LinearNDInterpolator
from scipy.interpolate import NearestNDInterpolator
from idwr import idw
from idwr import idwr
import numpy as np
from scipy.interpolate import interp2d
from scipy.interpolate import Rbf
from scipy.interpolate import SmoothBivariateSpline
import seaborn as sns

#Plot maps for the Texas dataset

arqle=pd.read_csv("texas.csv",names=None,header=None)
vec=arqle.to_numpy();

x=vec[:,0]
y=vec[:,1]
z=vec[:,2]

npt=50
dx=(max(x)-min(x))/npt
dy=(max(y)-min(y))/npt

xnew = np.arange(min(x), max(x),dx)
ynew = np.arange(min(y), max(y), dy)
xx, yy = np.meshgrid(xnew, ynew)

xx=xx.reshape(npt*npt,1)[:,0]
yy=yy.reshape(npt*npt,1)[:,0]

zidw=idw(x,y,z,xx,yy)
zidwr=idwr(x,y,z,xx,yy)
znew_idw=zidw.reshape(50,50)
znew_idwr=zidwr.reshape(50,50)

sns.heatmap(znew_idw)

sns.heatmap(znew_idwr)

rbfi = Rbf(x, y, z)  # radial basis function interpolator instance
zrbf = rbfi(xx, yy)   # interpolated values]
znew_rbf=zrbf.reshape(50,50)

sns.heatmap(znew_rbf)

points=np.zeros((len(x),2))
points[:,0]=x
points[:,1]=y 
bvsi=NearestNDInterpolator(points,z)
znn=bvsi(xx,yy)
znew_nn=znn.reshape(50,50)     
         
sns.heatmap(znew_nn)

bvs=SmoothBivariateSpline(x,y,z)
zbvs=bvs.ev(xnew,ynew)  
znew_bvs=zlin.reshape(50,50)     
         
sns.heatmap(znew_bvs) 

