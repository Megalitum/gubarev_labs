# Model reduction
This repository contains several algorithms implemented in Python using numpy/scipy for model order reduction.
## Variance method
This algorithm type can be described as tricky regressor, viz. it fits input system's response to function of a kind:
![$y(\delta k) = \sum\limits_{q=1}^Q (f_c^q \cos(\delta k) + f_s^q \sin(\delta k)) \exp(-\alpha \delta k)$](https://latex.codecogs.com/gif.latex?$y(\delta&space;k)&space;=&space;\sum\limits_{q=1}^Q&space;(f_c^q&space;\cos(\delta&space;k)&space;&plus;&space;f_s^q&space;\sin(\delta&space;k))&space;\exp(-\alpha&space;\delta&space;k)$)
## SVD method
This repository contains one of the most clean and verified implementation in IASA for model order reduction based on singular value decomposition.
