# Simpler-Grassmannians

This is the working code for the paper "Simpler Grassmannian optimization".

The code is written for sole purpose of conducting experiments of our paper and therefore we did not utilize any existing manifold optimization package (pymanopt for example). It is possible if I have used those existing packages, the code will be much cleaner and can be used for more general problems.

As a result, the current code is totally not satisfactory. The code use 4 different classes to solve 2 specific problems by 2 different framework. The formula derived in the paper only works for 2k<= n. So the code only works for the case 2k<= n.

If you have any problems. Please contact <laizehua@uchicago.edu>
