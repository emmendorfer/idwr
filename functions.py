# -*- coding: utf-8 -*-
"""
Some test functions.
    
 Those functions were adopted in:
 L. R. Emmendorfer, G. P. Dimuro, A point interpolation algorithm resulting
 from weighted linear regression (to appear)
    
 L. R. Emmendorfer, G. P. Dimuro, A novel formulation for 
 inverse distance weighting from weighted linear regression, in: 
 Computational Science - ICCS 2020 - 20th International Conference, 
 Amsterdam, The Netherlands, June 3-5, 2020, Proceedings, Part II, 
 Vol. 12138 of Lecture Notes in Computer 684 Science, Springer, 2020, 
 pp. 576-589. doi:10.1007/978-3-030-50417-5 43.
 URL https://doi.org/10.1007/978-3-030-50417-5 43

Created on Thu Dec 24 15:13:23 2020

@author: Leonardo Ramos Emmendorfer
"""

import math


def rosen(x1,x2):        
# range [-2.048, 2.048]
    return(100*(x2-x1**2)**2+(x1-1)**2)

def rast(x, y):
#range [-5.12..5.12]
    return (20+(x**2-10*math.cos(2*math.pi*x))+(y**2-10*math.cos(2*math.pi*y)))

def loggold(x, y):
#range [-2..2]
    return ((1/2.427)*(math.log((1+(x+y+1)**2*(19-14*x+3*x**2-14*y+6*x*y+3*y**2))*(30+(2*x-3*y)**2*(18-32*x+12*x**2+48*y-36*x*y+27*y**2)))-8.693))

def f102(x1, x2):
#range [-512, 512]
    term1 = -(x2+47) * math.sin(math.sqrt(abs(x2+x1/2+47)))
    term2 = -x1 * math.sin(math.sqrt(abs(x1-(x2+47))))
    return(term1 + term2)

def himm(x1,x2):
#range [-5,5]
    return((x1**2+x2-11)**2 + (x1+x2**2-7)**2)

def somb(x, y):
#range [0,1]
    if x==0.5 and y==0.5:
        s = 1
    if (x!= 0.5 or y!= 0.5):
        s = math.sin((16*(x-0.5))**2+(16*(y-0.5))**2)/((16*(x-0.5))**2+(16*(y-0.5))**2)
    return(s)
   