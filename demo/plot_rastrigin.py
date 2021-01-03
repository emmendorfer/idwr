# -*- coding: utf-8 -*-
"""
Created on Thu Dec 24 11:06:44 2020

@author: Leonardo Ramos Emmendorfer


This is the source code used to produce Fig.5 shown in:
 L. R. Emmendorfer, G. P. Dimuro, A point interpolation algorithm resulting
 from weighted linear regression (to appear)
"""

import numpy as np

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from functions import rast
from idwr import idw
from idwr import idwr
from scipy.interpolate import Rbf
from pykrige.ok import OrdinaryKriging


dif=1.12;

Xa = np.arange(-5.12, -4, 10.24/511)
Ya = np.arange(4, 5.12, 10.24/511)
res=len(Xa)

Zc=np.zeros(res**2,dtype=float)

X, Y = np.meshgrid(Xa, Ya)
Xc=X.reshape(res**2,1)
Yc=Y.reshape(res**2,1)

for i in range(res**2):
    Zc[i] = rast(Xc[i],Yc[i])
Z=Zc.reshape(res,res)
           
fig = plt.figure()
ax = fig.gca(projection='3d')

ax.set_zlim(0, 80)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False,vmin=12,vmax=80)
fig.colorbar(surf)
plt.savefig('rast_zoom.pdf', dpi=300)
plt.show()

z=np.zeros(6,dtype=float)

# in this version the input points are fixed to the values adopted in the paper
x=np.array([-5.05578847, -4.15479469, -4.62920184, -4.82670265, -4.89773737,
       -4.91703468])
y=np.array([4.39848489, 4.80091864, 4.78213963, 4.96204241, 4.062293  ,
       4.10969935])

#please uncomment below for other random points
#x=np.random.uniform(-5.12, -4, 6)   
#y=np.random.uniform(4, 5.12, 6)

for i in range(6):
    z[i]=rast(x[i],y[i])
 
zidw=idw(x,y,z,Xc[:,0],Yc[:,0])
zidwr=idwr(x,y,z,Xc[:,0],Yc[:,0])
                  
Zidw=zidw.reshape(res,res)
Zidwr=zidwr.reshape(res,res)
     
fig = plt.figure()
ax = fig.gca(projection='3d')

ax.set_zlim(0, 80)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
surf = ax.plot_surface(X, Y, Zidw, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False,vmin=12,vmax=80)
fig.colorbar(surf)
plt.savefig('rast_idw.pdf', dpi=300)
plt.show()
              
fig = plt.figure()
ax = fig.gca(projection='3d')

ax.set_zlim(0, 80)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
surf = ax.plot_surface(X, Y, Zidwr, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False,vmin=12,vmax=80)
fig.colorbar(surf)
plt.savefig('rast_idwr.pdf', dpi=300)
plt.show()



rbfi = Rbf(x, y, z)  # radial basis function interpolator instance
zrbf = (rbfi(Xc[:,0], Yc[:,0]))
Zrbf=zrbf.reshape(res,res)     

fig = plt.figure()
ax = fig.gca(projection='3d')

ax.set_zlim(0, 80)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
surf = ax.plot_surface(X, Y, Zrbf, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False,vmin=12,vmax=80)
fig.colorbar(surf)
plt.savefig('rast_rbf.pdf', dpi=300)
plt.show()



OKmodel = OrdinaryKriging(x,y,z,variogram_model="gaussian",verbose=False,enable_plotting=False)

zkrig, ss = OKmodel.execute("grid", X, Y)


Zkrig=np.zeros((res,res),dtype=float)
for i in range(res):
    for j in range(res):
        Zkrig[i,j]=zkrig[i,j]

fig = plt.figure()
ax = fig.gca(projection='3d')

ax.set_zlim(0, 80)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
surf = ax.plot_surface(X, Y, Zkrig, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False,vmin=12,vmax=80)
fig.colorbar(surf)
plt.savefig('rast_ok-g.pdf', dpi=300)
plt.show()



          


