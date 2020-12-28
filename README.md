# idwr
This project contains the source code of the IDWR interpolation algorithm and also the code used for its evaluation, as described in:

 L. R. Emmendorfer, G. P. Dimuro, A point interpolation algorithm resulting
 from weighted linear regression (to appear, 2021)
 
Besides REAEME.md, the following files are also included in the project:
 
Implementations of IDWR and IDW:
 idwr.py   
 
Code adopted for the evaluation of IDWR and other algorithms, as shown in the paper: 
 eval_functions.py
 eval_datasets.py
 comp_time_eval.py
 
Functions and datasets used in the evaluation:
 functions.py
 calabria.csv
 cretaceous.csv
 texas.csv
 amazon.csv
 
Code used for building some figures in the paper: 
 fig1.py   # used for figure 1
 plot_rastrigin.py  # used for figure 5
 
And a demonstration on how to generate maps from IDWR and other algorithms:
 plot_datasets.py
 
The IDWR was initially proposed in:

 L. R. Emmendorfer, G. P. Dimuro, A novel formulation for 
 inverse distance weighting from weighted linear regression, in: 
 Computational Science - ICCS 2020 - 20th International Conference, 
 Amsterdam, The Netherlands, June 3-5, 2020, Proceedings, Part II, 
 Vol. 12138 of Lecture Notes in Computer 684 Science, Springer, 2020, 
 pp. 576-589. doi:10.1007/978-3-030-50417-5 
 URL https://doi.org/10.1007/978-3-030-50417-5 



 
 
 
 
 
