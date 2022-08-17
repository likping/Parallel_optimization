# coding=utf-8
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np

class LGLModel:
    N = 0
    xn = None
    Pn = None
    w = None
    D = None

    def __init__(self, N):
        self.N = N
        self.xn, self.Pn,self.L= self.lglPoint(N)
        self.w = 2. / (N * (N + 1) * self.Pn ** 2)
        self.D = self.collocD(self.xn)

    def lglPoint(self, N):
        k = np.arange(0, N + 1)
        xn = np.cos(np.pi * k / N)
        xn = np.sort(xn)
        Pn, Pn_,L = self.LegendrFunction(N, xn)
        dx = (xn * Pn - Pn_) / ((N + 1) * Pn)
        keson = max(abs(dx))
        while keson > 1e-16:
            xn = xn - dx
            [Pn, Pn_,L] = self.LegendrFunction(N, xn)
            dx = (xn * Pn - Pn_) / ((N + 1) * Pn)
            keson = max(abs(dx))
        return xn, Pn, L

    def LegendrFunction(self, N, x):
        m = np.size(x)
        L = np.zeros((N + 1, m))
        L[0, :] = np.ones(m)
        L[1, :] = x
        # L[2, :] = (3 * x ** 2 - 1) / 2
        for n in range(1, N):
            L[n + 1, :] = ((2 * n + 1) * x * L[n, :] - n * L[n - 1, :]) / (n + 1)
        Ln = L[N, :]
        Ln_ = L[N - 1, :]
        return Ln, Ln_,L

    def collocD(self, x):
        N = len(x)
        N2 = N * N
        # Compute the barycentric weights
        X = np.tile(x.T, [N, 1]).T
        Xdiff = X - X.T + np.eye(N)
        temp = np.prod(Xdiff, axis=1)
        W = np.tile(1 / temp, [N, 1]).T
        D = W / (W.T * Xdiff)
        temp2 = 1 - sum(D)
        for i in np.arange(0, N):
            D[i, i] = temp2[i]
        D = -D.T
        return D

    def interpMatrix(self, taoPoint):
        m = len(taoPoint)
        N = self.N
        taoN = self.xn
        Ln = self.Pn
        Lnt, Lnt_,_= self.LegendrFunction(N, taoPoint)
        interpMatrix = np.zeros((N + 1, m))
        for l in range(N + 1):
            for i in range(m):
                if np.abs(taoPoint[i] - taoN[l]) < 1e-6:
                    interpMatrix[l, i] = 1
                else:
                    interpMatrix[l, i] = (taoPoint[i] * Lnt[i] - Lnt_[i]) / (
                            (taoPoint[i] - taoN[l]) * ((N + 1) * Ln[l]))
        return interpMatrix
    #计算Birkhoff矩阵
    def BirkhoffMatrix(self,W,N,L):
        B=np.zeros((N+1,N+1))
        Beta=np.zeros((N+1,N+1))
        Fai=np.zeros((N+1,N+1))
        B[:,0]=1
        B[0,1:]=0
        # B[N,:]=0
        for k in range (0,N):
            for j in range(1,N+1):
                Beta[k,j]=W[j]*(L[k,j]-((-1)**(N+k))*L[N,j])
        # sy.Matrix(Beta)   
        for k in range (0,N):
            for i in range(0,N+1):
                if k>=1:
                    Fai[k,i]=(1/(2*k+1))*(L[k+1,i]-L[k-1,i])/(2/(2*k+1))
                else:
                    Fai[k,i]=(L[k+1,i]+1)/(2/(2*k+1))
        # sy.Matrix(Fai)
        for i in range(0,N+1):
            for j in range(1,N+1):
                temp=0
                for k in range(0,N):
                    temp=temp+Beta[k,j]*Fai[k,i]
                B[i,j]=temp
        # B_m=sy.Matrix(B)
        return B