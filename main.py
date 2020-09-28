"""
This file includes the code for the figures in the papers.
"""
import numpy as np
from scipy import linalg
import time
import matplotlib
matplotlib.rcParams['text.usetex'] = True
import matplotlib.pyplot as plt
from grassopt import *


"""
Set the trace optimization model.
"""

np.random.seed(0)

m=6
n=10
A = np.diag(range(m+n))
test_model_1 = EAS_trace_opt(m, n, A)
Q_0 = random_orthogonal(m+n)
Y_0 = Q_0[:,:m]
w, v = linalg.eigh(A)
opt = sum(w[:m])
Y_opt = v[:,:m]
test_model_2 = Simpler_trace_opt(m, n, A)

Y_0 = test_model_1.EAS_steepest_gradient(Y_0, 20, orthogonalize = True)
Q_0 = test_model_2.simpler_steepest_gradient(Q_0, 20)

"""
This is the code for Figure 1. To get an orthogonalized version of EAS, add parameter "orthogonalize = True" in every EAS function.
"""

fig, (ax, ax1) = plt.subplots(ncols=2, constrained_layout=True, dpi=400)
plt.figure(dpi=1200)
ax.set_xlim(0, 100)
ax.set_yscale('log')
ax.set_title('Stiefel model')
ax.set_xticks(range(0, 101, 10))
ax.set_xlabel('iterations')
ax.set_ylabel(r'$\| Y_i Y_i^{\scriptscriptstyle\mathsf{T}} - Y_*Y_*^{\scriptscriptstyle\mathsf{T}} \|_{\scriptscriptstyle\mathsf{F}}$')

error_list = []
Y=Y_0
error_list.append(Fro_norm(Y,Y_opt))
for i in range(100):
    Y = test_model_1.EAS_steepest_gradient(Y, 1)
    error_list.append(Fro_norm(Y,Y_opt))
line, = ax.plot(range(101),error_list, label='GD')  


error_list = []
Y=Y_0
error_list.append(Fro_norm(Y,Y_opt))
Y, Ylist = test_model_1.EAS_BB_gradient(Y, 99)
for Y in Ylist:
    error_list.append(Fro_norm(Y,Y_opt)) 
line, = ax.plot(range(101),error_list, label='BB') 


error_list = []
Y=Y_0
error_list.append(Fro_norm(Y,Y_opt))
Y, Ylist = test_model_1.EAS_conjugate_gradient(Y, 100, m*n)
for Y in Ylist:
    error_list.append(Fro_norm(Y,Y_opt))
line, = ax.plot(range(101),error_list, label='CG') 

error_list = []
Y=Y_0
error_list.append(Fro_norm(Y,Y_opt))
for i in range(100):
    Y = test_model_1.EAS_Newton(Y, 1)
    error_list.append(Fro_norm(Y,Y_opt))
line, = ax.plot(range(101),error_list, label='NT') 
ax.legend()



ax1.set_title('Involution model')
ax1.set_xlim(0, 100)
ax1.set_yscale('log')
ax1.set_xticks(range(0, 101, 10))
ax1.set_xlabel('iterations')
ax1.set_ylabel(r'$\| Q_i - Q_* \|_{\scriptscriptstyle\mathsf{F}}$')

error_list = []
Q=Q_0
error_list.append(Fro_norm(Q[:,:m],Y_opt))
for i in range(100):
    Q = test_model_2.simpler_steepest_gradient(Q, 1)
    error_list.append(Fro_norm(Q[:,:m],Y_opt))
line, = ax1.plot(range(101),error_list, label='GD') 

error_list = []
Q=Q_0
error_list.append(Fro_norm(Q[:,:m],Y_opt))
Q, Qlist = test_model_2.simpler_BB_gradient(Q, 99)
for Q in Qlist:
    error_list.append(Fro_norm(Q[:,:m],Y_opt))
line, = ax1.plot(range(101),error_list, label='BB') 

error_list = []
Q=Q_0
error_list.append(Fro_norm(Q[:,:m],Y_opt))
Q, Qlist = test_model_2.simpler_conjugate_gradient(Q, 100, m*n)
for Q in Qlist:
    error_list.append(Fro_norm(Q[:,:m],Y_opt))
line, = ax1.plot(range(101),error_list, label='CG') 

error_list = []
Q=Q_0
error_list.append(Fro_norm(Q[:,:m],Y_opt))
for i in range(100):
    Q = test_model_2.simpler_Newton(Q, 1)
    error_list.append(Fro_norm(Q[:,:m],Y_opt))
line, = ax1.plot(range(101),error_list, label='NT') 
ax1.legend()

"""
This is the code for Figure 2.
"""


fig, (ax, ax1) = plt.subplots(ncols=2, constrained_layout=True,  dpi=400)
ax.set_xlim(0, 100)
ax.set_yscale('log')
ax.set_title('Stiefel model')
ax.set_xticks(range(0, 101, 10))
ax.set_xlabel('iterations')
ax.set_ylabel(r'$\| Y_i^{\scriptscriptstyle\mathsf{T}} Y_i - I\|_{\scriptscriptstyle\mathsf{F}}$')

error_list = []
Y=Y_0
error_list.append(linalg.norm(Y.T.dot(Y)-np.eye(m)))
for i in range(100):
    Y = test_model_1.EAS_steepest_gradient(Y, 1)
    error_list.append(linalg.norm(Y.T.dot(Y)-np.eye(m)))
line, = ax.plot(range(101),error_list, label='GD')  


error_list = []
Y=Y_0
error_list.append(linalg.norm(Y.T.dot(Y)-np.eye(m)))
Y, Ylist = test_model_1.EAS_BB_gradient(Y, 99)
for Y in Ylist:
    error_list.append(linalg.norm(Y.T.dot(Y)-np.eye(m))) 
line, = ax.plot(range(101),error_list, label='BB') 


error_list = []
Y=Y_0
error_list.append(linalg.norm(Y.T.dot(Y)-np.eye(m)))
Y, Ylist = test_model_1.EAS_conjugate_gradient(Y, 100, m*n)
for Y in Ylist:
    error_list.append(linalg.norm(Y.T.dot(Y)-np.eye(m)))
line, = ax.plot(range(101),error_list, label='CG') 



error_list = []
Y=Y_0
error_list.append(linalg.norm(Y.T.dot(Y)-np.eye(m)))
for i in range(100):
    Y = test_model_1.EAS_Newton(Y, 1)
    error_list.append(linalg.norm(Y.T.dot(Y)-np.eye(m)))
line, = ax.plot(range(101),error_list, label='NT') 
ax.legend()




ax1.set_xlim(0, 100)
ax1.set_yscale('log')
ax1.set_ylim(10**(-16), 10**(-12))
ax1.set_xticks(range(0, 101, 10))
ax1.set_xlabel('iterations')
ax1.set_ylabel(r'$\| Q_i^2 - I\|_{\scriptscriptstyle\mathsf{F}}$')
ax1.set_title('Involution model')

error_list = []
Q=Q_0
error_list.append(linalg.norm(Q.T.dot(Q)-np.eye(m+n)))
for i in range(100):
    Q = test_model_2.simpler_steepest_gradient(Q, 1)
    error_list.append(linalg.norm(Q.T.dot(Q)-np.eye(m+n)))
line, = ax1.plot(range(101),error_list, label='GD') 


error_list = []
Q=Q_0
error_list.append(linalg.norm(Q.T.dot(Q)-np.eye(m+n)))
Q, Qlist = test_model_2.simpler_BB_gradient(Q, 99)
for Q in Qlist:
    error_list.append(linalg.norm(Q.T.dot(Q)-np.eye(m+n)))
line, = ax1.plot(range(101),error_list, label='BB') 

 

error_list = []
Q=Q_0
error_list.append(linalg.norm(Q.T.dot(Q)-np.eye(m+n)))
Q, Qlist = test_model_2.simpler_conjugate_gradient(Q, 100, m*n)
for Q in Qlist:
    error_list.append(linalg.norm(Q.T.dot(Q)-np.eye(m+n)))
line, = ax1.plot(range(101),error_list, label='CG') 


error_list = []
Q=Q_0
error_list.append(linalg.norm(Q.T.dot(Q)-np.eye(m+n)))
for i in range(100):
    Q = test_model_2.simpler_Newton(Q, 1)
    error_list.append(linalg.norm(Q.T.dot(Q)-np.eye(m+n)))
line, = ax1.plot(range(101),error_list, label='NT') 
ax1.legend(loc = 'lower right')


"""
Set the Frechet mean model.
"""

np.random.seed(0)

m=6
n=10
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
This is the plot of Figure 3.
"""

fig, (ax, ax1) = plt.subplots(ncols=2, constrained_layout=True,  dpi=400)
ax.set_xlim(0, 100)
ax.set_yscale('log')
ax.set_xticks(range(0, 101, 10))
ax.set_xlabel('iterations')
ax.set_ylabel(r'$\| \nabla f(Y_i) \|_{\scriptscriptstyle\mathsf{F}}$')
ax.set_title('Stiefel model')

error_list = []
Y=Y_0
error_list.append(linalg.norm(test_model_1.EAS_gradient(Y)))
for i in range(100):
    Y = test_model_1.EAS_steepest_gradient(Y, 1)
    error_list.append(linalg.norm(test_model_1.EAS_gradient(Y)))
line, = ax.plot(range(101),error_list, label='GD')  


error_list = []
Y=Y_0
error_list.append(linalg.norm(test_model_1.EAS_gradient(Y)))
Y, Ylist = test_model_1.EAS_BB_gradient(Y, 99)
for Y in Ylist:
    error_list.append(linalg.norm(test_model_1.EAS_gradient(Y)))
line, = ax.plot(range(101),error_list, label='BB') 


error_list = []
Y=Y_0
error_list.append(linalg.norm(test_model_1.EAS_gradient(Y)))
Y, Ylist = test_model_1.EAS_conjugate_gradient(Y, 100, m*n)
for Y in Ylist:
    error_list.append(linalg.norm(test_model_1.EAS_gradient(Y)))
line, = ax.plot(range(101),error_list, label='CG') 
ax.legend(loc = 'right')


ax1.set_xlim(0, 100)
ax1.set_yscale('log')
ax1.set_xticks(range(0, 101, 10))
ax1.set_xlabel('iterations')
ax1.set_ylabel(r'$\| \nabla f(Q_i) \|_{\scriptscriptstyle\mathsf{F}}$')
ax1.set_title('Involution model')

error_list = []
Q=Q_0
error_list.append(linalg.norm(test_model_2.simpler_gradient(Q)))
for i in range(100):
    Q = test_model_2.simpler_steepest_gradient(Q, 1)
    error_list.append(linalg.norm(test_model_2.simpler_gradient(Q)))
line, = ax1.plot(range(101),error_list, label='GD') 

error_list = []
Q=Q_0
error_list.append(linalg.norm(test_model_2.simpler_gradient(Q)))
Q, Qlist = test_model_2.simpler_BB_gradient(Q, 99)
for Q in Qlist:
    error_list.append(linalg.norm(test_model_2.simpler_gradient(Q)))
line, = ax1.plot(range(101),error_list, label='BB') 
 

error_list = []
Q=Q_0
error_list.append(linalg.norm(test_model_2.simpler_gradient(Q)))
Q, Qlist = test_model_2.simpler_conjugate_gradient(Q, 100, m*n)
for Q in Qlist:
    error_list.append(linalg.norm(test_model_2.simpler_gradient(Q)))
line, = ax1.plot(range(101),error_list, label='CG')  

ax1.legend()


"""
This is the plot of Figure 4.
"""

fig, (ax, ax1) = plt.subplots(ncols=2, constrained_layout=True,  dpi=400)
ax.set_xlim(0, 100)
ax.set_yscale('log')
ax.set_xticks(range(0, 101, 10))
ax.set_xlabel('iterations')
ax.set_ylabel(r'$\| Y_i^{\scriptscriptstyle\mathsf{T}} Y_i - I\|_{\scriptscriptstyle\mathsf{F}}$')
ax.set_title('Stiefel model')


error_list = []
Y=Y_0
error_list.append(linalg.norm(Y.T.dot(Y)-np.eye(m)))
for i in range(100):
    Y = test_model_1.EAS_steepest_gradient(Y, 1)
    error_list.append(linalg.norm(Y.T.dot(Y)-np.eye(m)))
line, = ax.plot(range(101),error_list, label='GD')  

error_list = []
Y=Y_0
error_list.append(linalg.norm(Y.T.dot(Y)-np.eye(m)))
Y, Ylist = test_model_1.EAS_BB_gradient(Y, 99)
for Y in Ylist:
    error_list.append(linalg.norm(Y.T.dot(Y)-np.eye(m))) 
line, = ax.plot(range(101),error_list, label='BB') 

error_list = []
Y=Y_0
error_list.append(linalg.norm(Y.T.dot(Y)-np.eye(m)))
Y, Ylist = test_model_1.EAS_conjugate_gradient(Y, 100, m*n)
for Y in Ylist:
    error_list.append(linalg.norm(Y.T.dot(Y)-np.eye(m)))
line, = ax.plot(range(101),error_list, label='CG') 
ax.legend(loc = 'right')


ax1.set_xlim(0, 100)
ax1.set_yscale('log')
ax1.set_ylim(10**(-16), 10**(-12))
ax1.set_xticks(range(0, 101, 10))
ax1.set_xlabel('iterations')
ax1.set_ylabel(r'$\| Q_i^2 - I\|_{\scriptscriptstyle\mathsf{F}}$')
ax1.set_title('Involution model')


error_list = []
Q=Q_0
error_list.append(linalg.norm(Q.T.dot(Q)-np.eye(m+n)))
for i in range(100):
    Q = test_model_2.simpler_steepest_gradient(Q, 1)
    error_list.append(linalg.norm(Q.T.dot(Q)-np.eye(m+n)))
line, = ax1.plot(range(101),error_list, label='GD') 

error_list = []
Q=Q_0
error_list.append(linalg.norm(Q.T.dot(Q)-np.eye(m+n)))
Q, Qlist = test_model_2.simpler_BB_gradient(Q, 99)
for Q in Qlist:
    error_list.append(linalg.norm(Q.T.dot(Q)-np.eye(m+n)))
line, = ax1.plot(range(101),error_list, label='BB') 


error_list = []
Q=Q_0
error_list.append(linalg.norm(Q.T.dot(Q)-np.eye(m+n)))
Q, Qlist = test_model_2.simpler_conjugate_gradient(Q, 100,20)
for Q in Qlist:
    error_list.append(linalg.norm(Q.T.dot(Q)-np.eye(m+n)))
line, = ax1.plot(range(101),error_list, label='CG') 
ax1.legend(loc = 'right')



"""
This is the plot of function value of the two models. It does not mean much.
"""

fig, (ax, ax1) = plt.subplots(ncols=2, constrained_layout=True,  dpi=400)
ax.set_xlim(0, 100)
ax.set_xticks(range(0, 101, 10))
ax.set_ylim(2, 10)
ax.set_xlabel('iterations')
ax.set_ylabel(r'$f(Y_i)$')
ax.set_title('Stiefel model')

error_list = []
Y=Y_0
error_list.append(2*test_model_1.EAS_function(Y))
for i in range(100):
    Y = test_model_1.EAS_steepest_gradient(Y, 1)
    error_list.append(2*test_model_1.EAS_function(Y))
line, = ax.plot(range(101),error_list, label='GD')  


error_list = []
Y=Y_0
error_list.append(2*test_model_1.EAS_function(Y))
Y, Ylist = test_model_1.EAS_BB_gradient(Y, 99)
for Y in Ylist:
    error_list.append(2*test_model_1.EAS_function(Y))
line, = ax.plot(range(101),error_list, label='BB') 


error_list = []
Y=Y_0
error_list.append(2*test_model_1.EAS_function(Y))
Y, Ylist = test_model_1.EAS_conjugate_gradient(Y, 100, m*n)
for Y in Ylist:
    error_list.append(2*test_model_1.EAS_function(Y))
line, = ax.plot(range(101),error_list, label='CG') 
ax.legend()


ax1.set_xlim(0, 100)
ax1.set_xticks(range(0, 101, 10))

ax1.set_xlabel('iterations')
ax1.set_ylabel(r'$f(Q_i)$')
ax1.set_title('Involution model')

error_list = []
Q=Q_0
error_list.append(test_model_2.simpler_function(Q))
for i in range(100):
    Q = test_model_2.simpler_steepest_gradient(Q, 1)
    error_list.append(test_model_2.simpler_function(Q))
line, = ax1.plot(range(101),error_list, label='GD') 


error_list = []
Q=Q_0
error_list.append(test_model_2.simpler_function(Q))
Q, Qlist = test_model_2.simpler_BB_gradient(Q, 99)
for Q in Qlist:
    error_list.append(test_model_2.simpler_function(Q))
line, = ax1.plot(range(101),error_list, label='BB') 
 

error_list = []
Q=Q_0
error_list.append(test_model_2.simpler_function(Q))
Q, Qlist = test_model_2.simpler_conjugate_gradient(Q, 100, m*n)
for Q in Qlist:
    error_list.append(test_model_2.simpler_function(Q))
line, = ax1.plot(range(101),error_list, label='CG') 
ax1.legend(loc = 'right')


    