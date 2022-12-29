# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 11:37:10 2020

@author: Leonardo Ramos Emmendorfer
"""

import numpy as np 


def idwr(x,y,z,xnew,ynew):
    '''Computes the interpolation from input points using the IDWR method.'''

# The IDWR interpolation method is proposed in:
# L. R. Emmendorfer, G. P. Dimuro, A novel formulation for 
# inverse distance weighting from weighted linear regression, in: 
# Computational Science - ICCS 2020 - 20th International Conference, 
# Amsterdam, The Netherlands, June 3-5, 2020, Proceedings, Part II, 
# Vol. 12138 of Lecture Notes in Computer 684 Science, Springer, 2020, 
# pp. 576-589. doi:10.1007/978-3-030-50417-5 43.
# URL https://doi.org/10.1007/978-3-030-50417-5 43

# Usage: idwr(x,y,z,xnew,ynew)

# The input, known points, are given as three arrays (x,y,z)
# The new locations are given in (xnew,ynew)
    

    n=len(x)  # number of data points (input)
    m=len(xnew)  #number of data points (output)
    
    znew_idwr=np.zeros(m, dtype=float)
    
    for j in range(m):   # compute all new points 
        dist=np.sqrt(np.add(np.power(x-xnew[j],2), np.power(y-ynew[j],2)) )  #Euclidean distance
        w=dist**(-2)
        s=1/np.sum(w)
        ws=w*s   # Weights
        sumz=np.sum(z)
        den=n**2-np.sum(w)*np.sum(dist**2)
        znew_idw=np.sum(np.multiply(ws,z))
        znew_idwr[j]=znew_idw+n*(sumz-n*znew_idw)/den       
    return znew_idwr


def idw(x,y,z,xnew,ynew):
    '''Computes the interpolation from input points using the original IDW method.'''

# The IDW interpolation method. It was used in:
# L. R. Emmendorfer, G. P. Dimuro, A novel formulation for 
# inverse distance weighting from weighted linear regression, in: 
# Computational Science - ICCS 2020 - 20th International Conference, 
# Amsterdam, The Netherlands, June 3-5, 2020, Proceedings, Part II, 
# Vol. 12138 of Lecture Notes in Computer 684 Science, Springer, 2020, 
# pp. 576-589. doi:10.1007/978-3-030-50417-5 43.
# URL https://doi.org/10.1007/978-3-030-50417-5 43

# Usage: idw(x,y,z,xnew,ynew)

# The input, known points, are given as three arrays (x,y,z)
# The new locations are given in (xnew,ynew)
    

    m=len(xnew)  #number of data points (output)
    
    znew_idw=np.zeros(m, dtype=float)
    
    for j in range(m):   # compute all new points 
        dist=np.sqrt(np.add(np.power(x-xnew[j],2), np.power(y-ynew[j],2)) )  #Euclidean distance 
        w=dist**(-2)
        s=1/np.sum(w)
        ws=w*s   # Weights
        znew_idw[j]=np.sum(np.multiply(ws,z))   
    return znew_idw

