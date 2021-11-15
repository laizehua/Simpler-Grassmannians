"""
This is the working code for the paper "Simpler Grassmannian optimization".

The code is written for sole purpose of conducting experiments of our paper and therefore we did not utilize any existing manifold optimization package (pymanopt for example). It is possible if I have used those existing packages, the code will be much cleaner and can be used for more general problems.

As a result, the current code is totally not satisfactory. The code use 4 different classes to solve 2 specific problems by 2 different framework. The formula derived in the paper only works for 2k<=n. So the code only works for the case 2k<=n.
"""
import numpy as np
from scipy import linalg

def orthogonalized(Y):
    q, r =  linalg.qr(Y, mode='economic')
    return q

def Fro_norm(Y_1, Y_2):
    return linalg.norm(Y_1.dot(Y_1.T)-Y_2.dot(Y_2.T))

def random_orthogonal(dimension):
    # random orthogonal matrix
    a = np.random.rand(dimension,dimension)
    q, r = linalg.qr(a)
    return q

def EAS_log(X, Y):
    ytx = Y.T.dot(X)
    At = Y.T - ytx.dot(X.T)
    Bt = np.linalg.solve(ytx, At)
    u, s, vt = linalg.svd(Bt.T, full_matrices=False)

    U = (u * np.arctan(s)).dot(vt)
    return U

def distance(X, Y):
    u, s, vt = linalg.svd(X.T.dot(Y))
    return linalg.norm(np.arccos(np.minimum(np.abs(s),np.ones(s.size))))


class EAS_trace_opt:
    # EAS algorithm for trace optimization
    def __init__(self, dimension1, dimension2, A_0):
        # Initialize by _m = k, _n = n-k and A
        self._m = dimension1
        self._n = dimension2
        self._A = A_0

    def svd_compact(self, H):
        U, s, V =  linalg.svd(H, full_matrices=False)
        U = U[:,:self._m]
        s = s[:self._m]
        V = V[:self._m,:]
        return U, s, V

    def EAS_geodesic_svd(self, Y, U, s, V):
        # Use SVD. Speed up the line search step a little bit.
        return (Y.dot(V.T)*(np.cos(s))).dot(V)+\
            (U*(np.sin(s))).dot(V)
 
    def EAS_transport_svd(self, Y, U, s, V, D):
        
        return (Y.dot(V.T)*(-np.sin(s))).dot(U.T).dot(D)+\
                (U*(np.cos(s))).dot(U.T).dot(D)+D-U.dot(U.T).dot(D)

    def EAS_armijo_linesearch_svd(self, Y, U, s, V):
        S = U.dot(np.diag(s)).dot(V)
        a = np.sum(np.multiply(self.EAS_gradient(Y),S))
        if a>0:
            return s*0, Y
        for i in np.arange(20, 0, -2, dtype = float):
            Y_new = self.EAS_geodesic_svd(Y, U, s*(2**(i-10)), V)
            if self.EAS_function(Y_new)-self.EAS_function(Y)< a*0.01*(2**(i-10)):
                return s*(2**(i-10)), Y_new
        return s*0, Y
    
    def EAS_gradient(self, Y):
        B = self._A.dot(Y)
        return B-Y.dot(Y.T).dot(B)

    def EAS_Newton_step(self, Y):
        Z = linalg.solve_sylvester(self._A, -Y.T.dot(self._A).dot(Y), Y)
        return -Y+Z.dot(linalg.inv(Y.T.dot(Z)))

    def EAS_error(self, Y):
        return linalg.norm(Y.dot(Y.T)-self.Y_opt.dot(self.Y_opt.T))

    def EAS_function(self, Y):
        return np.sum(np.multiply(Y,self._A.dot(Y)))/2

    def EAS_steepest_gradient(self, Y_0, t, orthogonalize = False):
        Y=Y_0
        for i in range(t):
            U, s, V = self.svd_compact(-self.EAS_gradient(Y))
            j, Y = self.EAS_armijo_linesearch_svd(Y, U, s, V)
            if orthogonalize == True:
                Y = orthogonalized(Y)
        return Y

    def EAS_BB_gradient(self, Y_0, t, orthogonalize = False):
        Y=Y_0
        G_old = self.EAS_gradient(Y)
        S = -G_old
        U, s, V =  self.svd_compact(S)
        G_old = self.EAS_transport_svd(Y, U, s, V, G_old)
        Y = self.EAS_geodesic_svd(Y, U, s, V)
        for i in range(t):
            G_new = self.EAS_gradient(Y)
            alpha = np.sum(np.multiply(G_new-G_old,S))/np.sum(np.multiply(G_new-G_old,G_new-G_old))
            S = -alpha*G_new
            U, s, V =  self.svd_compact(S)
            G_old = self.EAS_transport_svd(Y, U, s, V, G_new)
            Y = self.EAS_geodesic_svd(Y, U, s, V)
            if orthogonalize == True:
                Y = orthogonalized(Y)
        return Y

    def EAS_conjugate_gradient(self, Y_0, t, reset, orthogonalize = False):
        # Reset the conjugate gradient to the true gradient after $reset$ steps.
        Y_old = Y_0
        G_old = self.EAS_gradient(Y_old)
        P = -G_old
        for i in range(t):
            U, s, V =  self.svd_compact(P)
            s, Y_new = self.EAS_armijo_linesearch_svd(Y_old, U, s, V)
            if orthogonalize == True:
                Y_new = orthogonalized(Y_new)
            G_old = self.EAS_transport_svd(Y_old, U, s, V, G_old)
            P = self.EAS_transport_svd(Y_old, U, s, V, P)
            G_new = self.EAS_gradient(Y_new)
            gamma = np.sum(np.multiply(G_new-G_old,G_new))/np.sum(np.multiply(G_old,G_old))
            P = -G_new+gamma*P
            G_old = G_new
            Y_old = Y_new      
            if i%reset == reset-1:
                P=-G_new
        return Y_new
    
    def EAS_Newton(self, Y_0, t, orthogonalize = False):
        Y = Y_0
        for i in range(t):
            U, s, V = self.svd_compact(self.EAS_Newton_step(Y))
            Y = self.EAS_geodesic_svd(Y, U, s, V)
            if orthogonalize == True:
                Y = orthogonalized(Y)
        return Y

class Simpler_trace_opt:
    # Simpler algorithm for trace optimization
    def __init__(self, dimension1, dimension2, A_0):
        self._m = dimension1
        self._n = dimension2
        self._A = A_0
        
    def simpler_geodesic_svd(self, Q, U, s, V):
        L = np.block([[U, np.zeros((self._m, self._n))],
                       [np.zeros((self._n, self._m)),V.T]])
        D = np.block([[np.diag(np.cos(s/2)), -np.diag(np.sin(s/2)), np.zeros((self._m, self._n-self._m))],
                       [np.diag(np.sin(s/2)), np.diag(np.cos(s/2)), np.zeros((self._m, self._n-self._m))],
                       [np.zeros((self._n-self._m, 2*self._m)), np.eye(self._n-self._m,self._n-self._m)]])
        return Q.dot(L).dot(D).dot(L.T)

    def simpler_function(self, Q):
        return 2*np.sum(np.multiply((Q[:,:self._m]).dot((Q[:,:self._m]).T),self._A))

    def simpler_gradient(self, Q):
        return (Q.T.dot(self._A).dot(Q))[:self._m,self._m:]

    def simpler_Newton_step(self, Q):
        U = Q.T.dot(self._A).dot(Q)
        return linalg.solve_sylvester(U[:self._m,:self._m],-U[self._m:,self._m:],2*U[:self._m,self._m:])

    def simpler_armijo_linesearch_svd(self, Q, U, s, V):
        
        S = U.dot(np.diag(s)).dot(V[:self._m,:])
        a = np.sum(np.multiply(self.simpler_gradient(Q),S))
        if a>0:
            return s*0, Q
        for i in np.arange(20, 0, -2, dtype = float):
            Q_new = self.simpler_geodesic_svd(Q, U, s*(2**(i-10)), V)
            if self.simpler_function(Q_new)-self.simpler_function(Q)< a*0.01*(2**(i-10)):
                return s*(2**(i-10)), Q_new
        return s*0, Q

    def simpler_steepest_gradient(self, Q_0, t):
        Q=Q_0
        for i in range(t):
            B = self.simpler_gradient(Q)
            U, s, V =  linalg.svd(-B)
            s, Q = self.simpler_armijo_linesearch_svd(Q, U, s, V)
        return Q

    def simpler_BB_gradient(self, Q_0, t):
        Q=Q_0
        G_old = self.simpler_gradient(Q)
        S = -G_old
        U, s, V =  linalg.svd(S)
        Q = self.simpler_geodesic_svd(Q, U, s, V)
        for i in range(t):
            G_new = self.simpler_gradient(Q)
            alpha = np.sum(np.multiply(G_new-G_old,S))/np.sum(np.multiply(G_new-G_old,G_new-G_old))
            S = -alpha*G_new
            U, s, V =  linalg.svd(S)
            G_old = G_new
            Q = self.simpler_geodesic_svd(Q, U, s, V)
        return Q
    
    def simpler_conjugate_gradient(self, Q_0, t, reset):
        Q_old = Q_0
        G_old = self.simpler_gradient(Q_old)
        P = -G_old
        for i in range(t):
            U, s, V =  linalg.svd(P)
            s, Q_new = self.simpler_armijo_linesearch_svd(Q_old, U, s, V)
            G_new = self.simpler_gradient(Q_new)
            gamma = np.sum(np.multiply(G_new-G_old,G_new))/np.sum(np.multiply(G_old,G_old))
            P = -G_new+gamma*P
            G_old = G_new
            Q_old = Q_new 
            if i%reset == reset-1:
                P=-G_new
        return Q_new

    def simpler_Newton(self, Q_0, t):
        Q = Q_0
        for i in range(t):
            U, s, V = linalg.svd(self.simpler_Newton_step(Q))
            Q = self.simpler_geodesic_svd(Q, U, s, V)
        return Q

    def simpler_LBFGS(self, Q_0, t, k):
        Q = Q_0
        Ylist = [1]
        Slist = []
        Qlist = []
        alpha = np.zeros(k)
        beta = np.zeros(k)
        for i in range(t):
            G_new = self.simpler_gradient(Q)
            if i==0:
                S = -G_new
            elif i<k:
                S = -G_new
                Ylist.append(G_new-G_old)
            else:
                Ylist.append(G_new-G_old)
                Ylist.pop(0)
                P=G_new
                for j in range(k-1,-1,-1):
                    alpha[j] = np.sum(np.multiply(Slist[j],P))/np.sum(np.multiply(Slist[j],Ylist[j]))
                    P -= alpha[j]*Ylist[j]
                Z = np.sum(np.multiply(Ylist[k-1],Slist[k-1]))/np.sum(np.multiply(Ylist[k-1],Ylist[k-1]))*P
                for j in range(k):
                    beta[j] = np.sum(np.multiply(Ylist[j],Z))/np.sum(np.multiply(Slist[j],Ylist[j]))
                    Z += (alpha[j]-beta[j])*Slist[j]
                S = -Z
                Slist.pop(0)
            U, s, V =  linalg.svd(S)
            s = s/max(linalg.norm(s), 1)
            s, Q = self.simpler_wolfe_linesearch_svd(Q, U, s, V)
            S = U.dot(np.diag(s)).dot(V[:self._m,:])
            Slist.append(S)
            G_old = G_new
            Qlist.append(Q)
        return Q, Qlist

class EAS_Frechet_opt:
    # EAS algorithm for Frechet mean
    def __init__(self, dimension1, dimension2, Alist, length):
        # Initialize by _m = k, _n = n-k, a list of orthogonal matrices, and the length of the list.
        self._m = dimension1
        self._n = dimension2
        self._Alist = Alist
        self._length = length

    def svd_compact(self, H):
        U, s, V =  linalg.svd(H, full_matrices=False)
        U = U[:,:self._m]
        s = s[:self._m]
        V = V[:self._m,:]
        return U, s, V
        
    def EAS_gradient(self, Y):
        G = np.zeros((self._m+self._n, self._m))
        for i in range(self._length):
            G-= EAS_log(Y, self._Alist[i])
        return G

    def EAS_function(self, Y):
        f=0
        for i in range(self._length):
            f+= distance(Y, self._Alist[i])**2/2
        return f
    
    def EAS_geodesic_svd(self, Y, U, s, V):
        return Y.dot(V.T).dot(np.diag(np.cos(s))).dot(V)+\
            U.dot(np.diag(np.sin(s))).dot(V)
 
    def EAS_transport_svd(self, Y, U, s, V, D):
        
        return Y.dot(V.T).dot(np.diag(-np.sin(s))).dot(U.T).dot(D)+\
                U.dot(np.diag(np.cos(s))).dot(U.T).dot(D)+D-U.dot(U.T).dot(D)
      
    def EAS_armijo_linesearch_svd(self, Y, U, s, V):
        S = U.dot(np.diag(s)).dot(V)
        a = np.sum(np.multiply(self.EAS_gradient(Y),S))
        if a>0:
            return s*0, Y
        for i in np.arange(20, 0, -2, dtype = float):
            Y_new = self.EAS_geodesic_svd(Y, U, s*(2**(i-10)), V)
            if self.EAS_function(Y_new)-self.EAS_function(Y)< a*0.01*(2**(i-10)):
                return s*(2**(i-10)), Y_new
        return s*0, Y

    def EAS_steepest_gradient(self, Y_0, t, orthogonalize = False):
        Y=Y_0
        for i in range(t):
            U, s, V = self.svd_compact(-self.EAS_gradient(Y))
            j, Y = self.EAS_armijo_linesearch_svd(Y, U, s, V)
            if orthogonalize == True:
                Y = orthogonalized(Y)
        return Y

    def EAS_BB_gradient(self, Y_0, t, orthogonalize = False):
        Y=Y_0
        G_old = self.EAS_gradient(Y)
        S = -G_old
        U, s, V =  self.svd_compact(S)
        G_old = self.EAS_transport_svd(Y, U, s, V, G_old)
        Y = self.EAS_geodesic_svd(Y, U, s, V)
        for i in range(t):
            G_new = self.EAS_gradient(Y)
            alpha = np.sum(np.multiply(G_new-G_old,S))/np.sum(np.multiply(G_new-G_old,G_new-G_old))
            S = -alpha*G_new
            U, s, V =  self.svd_compact(S)
            G_old = self.EAS_transport_svd(Y, U, s, V, G_new)
            Y = self.EAS_geodesic_svd(Y, U, s, V)
            if orthogonalize == True:
                Y = orthogonalized(Y)
        return Y

    def EAS_conjugate_gradient(self, Y_0, t, reset, orthogonalize = False):
        Y_old = Y_0
        G_old = self.EAS_gradient(Y_old)
        P = -G_old
        for i in range(t):
            U, s, V =  self.svd_compact(P)
            s, Y_new = self.EAS_armijo_linesearch_svd(Y_old, U, s, V)
            if orthogonalize == True:
                Y_new = orthogonalized(Y_new)
            G_old = self.EAS_transport_svd(Y_old, U, s, V, G_old)
            P = self.EAS_transport_svd(Y_old, U, s, V, P)
            G_new = self.EAS_gradient(Y_new)
            gamma = np.sum(np.multiply(G_new-G_old,G_new))/np.sum(np.multiply(G_old,G_old))
            P = -G_new+gamma*P
            G_old = G_new
            Y_old = Y_new      
            if i%reset == reset-1:
                P=-G_new
        return Y_new
    
    def EAS_Newton(self, Y_0, t, orthogonalize = False):
        Y = Y_0
        for i in range(t):
            U, s, V = self.svd_compact(self.EAS_Newton_step(Y))
            Y = self.EAS_geodesic_svd(Y, U, s, V)
            if orthogonalize == True:
                Y = orthogonalized(Y)
        return Y

class Simpler_Frechet_opt:
    # Simpler algorithm for Frechet mean
    def __init__(self, dimension1, dimension2, Alist, length):
        self._m = dimension1
        self._n = dimension2
        self._Alist = Alist
        self._len = length
        
    def simpler_geodesic_svd(self, Q, U, s, V):
        L = np.block([[U, np.zeros((self._m, self._n))],
                       [np.zeros((self._n, self._m)),V.T]])
        D = np.block([[np.diag(np.cos(s/2)), -np.diag(np.sin(s/2)), np.zeros((self._m, self._n-self._m))],
                       [np.diag(np.sin(s/2)), np.diag(np.cos(s/2)), np.zeros((self._m, self._n-self._m))],
                       [np.zeros((self._n-self._m, 2*self._m)), np.eye(self._n-self._m,self._n-self._m)]])
        return Q.dot(L).dot(D).dot(L.T)

    def log(self, X, Y):
        A = X.T.dot(Y[:,:self._m])
        Vt,s,Ut = linalg.svd(A[self._m:,:].dot(linalg.inv(A[:self._m,:])))
        U = Ut.T
        V = Vt.T
        return (2*U*np.arctan(s)).dot(V[:self._m,:])

    def distance(self, X, Y):
        u, s, vt = linalg.svd(X[:,:self._m].T.dot(Y[:,:self._m]))
        return (linalg.norm(np.arccos(np.minimum(s,np.ones(self._m)))))**2
    
    def simpler_gradient(self, Q):
        G = np.zeros((self._m, self._n))
        for i in range(self._len):
            G-= self.log(Q, self._Alist[i])
        return G

    def simpler_function(self, Y):
        f=0
        for i in range(self._len):
            f+= self.distance(Y, self._Alist[i])
        return f
    
    def simpler_armijo_linesearch_svd(self, Q, U, s, V):
    
        S = U.dot(np.diag(s)).dot(V[:self._m,:])
        a = np.sum(np.multiply(self.simpler_gradient(Q),S))
        if a>0:
            return s*0, Q
        for i in np.arange(20, 0, -2, dtype = float):
            Q_new = self.simpler_geodesic_svd(Q, U, s*(2**(i-10)), V)
            if self.simpler_function(Q_new)-self.simpler_function(Q)< a*0.01*(2**(i-10)):
                return s*(2**(i-10)), Q_new
        return s*0, Q

    def simpler_steepest_gradient(self, Q_0, t, steplength = 2):
        Q=Q_0
        for i in range(t):
            B = self.simpler_gradient(Q)
            U, s, V =  linalg.svd(-B)
            s, Q = self.simpler_armijo_linesearch_svd(Q, U, s, V)
        return Q
    
    def simpler_fix_gradient(self, Q_0, t):
        Q=Q_0
        for i in range(t):
            B = self.simpler_gradient(Q)
            U, s, V =  linalg.svd(-B)
            Q = self.simpler_geodesic_svd(Q, U, s, V)
        return Q

    def simpler_BB_gradient(self, Q_0, t):
        Q=Q_0
        G_old = self.simpler_gradient(Q)
        S = -G_old
        U, s, V =  linalg.svd(S)
        Q = self.simpler_geodesic_svd(Q, U, s, V)
        for i in range(t):
            G_new = self.simpler_gradient(Q)
            alpha = np.sum(np.multiply(G_new-G_old,S))/np.sum(np.multiply(G_new-G_old,G_new-G_old))
            S = -alpha*G_new
            U, s, V =  linalg.svd(S)
            G_old = G_new
            Q = self.simpler_geodesic_svd(Q, U, s, V)
        return Q

    def simpler_conjugate_gradient(self, Q_0, t, reset):
        Q_old = Q_0
        G_old = self.simpler_gradient(Q_old)
        P = -G_old
        for i in range(t):
            U, s, V =  linalg.svd(P)
            s, Q_new = self.simpler_armijo_linesearch_svd(Q_old, U, s, V)
            G_new = self.simpler_gradient(Q_new)
            gamma = np.sum(np.multiply(G_new-G_old,G_new))/np.sum(np.multiply(G_old,G_old))
            P = -G_new+gamma*P
            G_old = G_new
            Q_old = Q_new 
            if i%reset == reset-1:
                P=-G_new
        return Q_new

    def simpler_Newton(self, Q_0, t):
        Q = Q_0
        for i in range(t):
            U, s, V = linalg.svd(self.simpler_Newton_step(Q))
            Q = self.simpler_geodesic_svd(Q, U, s, V)
        return Q

    def simpler_LBFGS(self, Q_0, t, k):
        Q = Q_0
        Ylist = [1]
        Slist = []
        Qlist = []
        alpha = np.zeros(k)
        beta = np.zeros(k)
        for i in range(t):
            G_new = self.simpler_gradient(Q)
            if i==0:
                S = -G_new
            elif i<k:
                S = -G_new
                Ylist.append(G_new-G_old)
            else:
                Ylist.append(G_new-G_old)
                Ylist.pop(0)
                P=G_new
                for j in range(k-1,-1,-1):
                    alpha[j] = np.sum(np.multiply(Slist[j],P))/np.sum(np.multiply(Slist[j],Ylist[j]))
                    P -= alpha[j]*Ylist[j]
                Z = np.sum(np.multiply(Ylist[k-1],Slist[k-1]))/np.sum(np.multiply(Ylist[k-1],Ylist[k-1]))*P
                for j in range(k):
                    beta[j] = np.sum(np.multiply(Ylist[j],Z))/np.sum(np.multiply(Slist[j],Ylist[j]))
                    Z += (alpha[j]-beta[j])*Slist[j]
                S = -Z
                Slist.pop(0)
            U, s, V =  linalg.svd(S)
            s = s/max(linalg.norm(s), 1)
            s, Q = self.simpler_wolfe_linesearch_svd(Q, U, s, V)
            S = U.dot(np.diag(s)).dot(V[:self._m,:])
            Slist.append(S)
            G_old = G_new
            Qlist.append(Q)
        return Q, Qlist

    
    
    
    
    