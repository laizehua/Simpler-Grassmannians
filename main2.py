"""
This file includes the code for the figures in the papers.
"""
import numpy as np
from scipy import linalg
import time
import matplotlib
matplotlib.rcParams['text.usetex'] = True
import matplotlib.pyplot as plt
from grassopt2 import *


"""
Set the trace optimization model.
"""

np.random.seed(0)

m=10
n=490
A = np.diag(range(m+n))
test_model_1 = EAS_trace_opt(m, n, A)
Q_0 = random_orthogonal(m+n)
Y_0 = Q_0[:,:m]
w, v = linalg.eigh(A)
opt = sum(w[:m])
Y_opt = v[:,:m]
test_model_2 = Simpler_trace_opt(m, n, A)

Y_0 = test_model_1.EAS_steepest_gradient(Y_0, 40, orthogonalize = True)
Q_0 = test_model_2.simpler_steepest_gradient(Q_0, 40)

"""
This is the code for Figure 1. To get an orthogonalized version of EAS, add parameter "orthogonalize = True" in every EAS function.
"""


error_list = []
Y=Y_0
error_list.append(Fro_norm(Y,Y_opt))
t1 = time.time()
Y = test_model_1.EAS_steepest_gradient(Y, 100, orthogonalize = True)
t2 = time.time()
print(t2-t1)


error_list = []
Y=Y_0
error_list.append(Fro_norm(Y,Y_opt))
t1 = time.time()
Y = test_model_1.EAS_BB_gradient(Y, 99, orthogonalize = True)
t2 = time.time()
print(t2-t1)



error_list = []
Y=Y_0
error_list.append(Fro_norm(Y,Y_opt))
t1 = time.time()
Y = test_model_1.EAS_conjugate_gradient(Y, 100, 5, orthogonalize = True)
t2 = time.time()
print(t2-t1)


Y=Y_0
error_list.append(Fro_norm(Y,Y_opt))
t1 = time.time()
Y = test_model_1.EAS_Newton(Y, 100, orthogonalize = True)
t2 = time.time()
print(t2-t1)




error_list = []
Q=Q_0
error_list.append(Fro_norm(Q[:,:m],Y_opt))
t1 = time.time()
Q = test_model_2.simpler_steepest_gradient(Q, 100)
t2 = time.time()
print(t2-t1)

error_list = []
Q=Q_0
error_list.append(Fro_norm(Q[:,:m],Y_opt))

t1 = time.time()

Q = test_model_2.simpler_BB_gradient(Q, 99)

t2 = time.time()
print(t2-t1)

error_list = []
Q=Q_0
error_list.append(Fro_norm(Q[:,:m],Y_opt))
t1 = time.time()
Q = test_model_2.simpler_conjugate_gradient(Q, 100, 5)
t2 = time.time()
print(t2-t1)

error_list = []
Q=Q_0
error_list.append(Fro_norm(Q[:,:m],Y_opt))
t1 = time.time()
Q = test_model_2.simpler_Newton(Q, 100)
t2 = time.time()
print(t2-t1)

"""
Set the Frechet mean model.
"""

np.random.seed(0)

Alist = []
Alist2 = []
A = random_orthogonal(m+n)
Alist.append(A[:,:m])
Alist2.append(A)
B = random_orthogonal(m+n)
Alist.append(B[:,:m])
Alist2.append(B)
C = random_orthogonal(m+n)
Alist.append(C[:,:m])
Alist2.append(C)

Q_0 = random_orthogonal(m+n)
Y_0 = Q_0[:,:m]
test_model_1 = EAS_Frechet_opt(m, n, Alist, 3)
test_model_2 = Simpler_Frechet_opt(m, n, Alist2, 3)

D = A.T.dot(B[:,:m])
U,s,V = linalg.svd(A[:m,:m].dot(A[m:,:m].T))
s = np.arcsin(s)
L = np.block([[U, np.zeros((m, n))],
                       [np.zeros((n, m)),V.T]])
D = np.block([[np.diag(np.cos(s/2)), -np.diag(np.sin(s/2)), np.zeros((m, n-m))],
                       [np.diag(np.sin(s/2)), np.diag(np.cos(s/2)), np.zeros((m, n-m))],
                       [np.zeros((n-m, 2*m)), np.eye(n-m,n-m)]])
Q_opt = A.dot(L).dot(D).dot(L.T)
Y_opt = Q_opt[:,:m]

Y_0 = test_model_1.EAS_steepest_gradient(Y_0, 10, orthogonalize=True)
Q_0 = test_model_2.simpler_steepest_gradient(Q_0, 10)


"""
This is the plot of Figure 2.
"""



Y=Y_0
error_list.append(linalg.norm(test_model_1.EAS_gradient(Y)))
t1 = time.time()
Y = test_model_1.EAS_steepest_gradient(Y, 100, orthogonalize = True)
t2 = time.time()
print(t2-t1)


Y=Y_0
error_list.append(linalg.norm(test_model_1.EAS_gradient(Y)))
t1 = time.time()
Y = test_model_1.EAS_BB_gradient(Y, 99, orthogonalize = True)
t2 = time.time()
print(t2-t1)


Y=Y_0
t1 = time.time()
Y = test_model_1.EAS_conjugate_gradient(Y, 100, 5, orthogonalize = True)
t2 = time.time()
print(t2-t1)



error_list = []
Q=Q_0
error_list.append(linalg.norm(test_model_2.simpler_gradient(Q)))
t1 = time.time()
Q = test_model_2.simpler_steepest_gradient(Q, 100)

t2 = time.time()
print(t2-t1)

error_list = []
Q=Q_0
t1 = time.time()
Q = test_model_2.simpler_BB_gradient(Q, 99)
t2 = time.time()
print(t2-t1)



Q=Q_0
error_list.append(linalg.norm(test_model_2.simpler_gradient(Q)))
t1 = time.time()
Q = test_model_2.simpler_conjugate_gradient(Q, 100, 5)
t2 = time.time()
print(t2-t1)



    