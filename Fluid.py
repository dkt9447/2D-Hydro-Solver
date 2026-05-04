import numpy as np
import matplotlib.pyplot as plt
#1D solver
class Hydro:
    def __init__(self, U_init, gamma=1.4):
        self.gamma=gamma
        self.U=U_init
        self.n=U_init[0].size
        self.primitive=np.empty(U_init.shape)
        self.F=np.empty(U_init.shape)
        self.U_inverse()
        self.F_transform()
        
    def U_inverse(self):
        U=self.U
        self.primitive[0]=U[0]
        self.primitive[1]=U[1]/U[0]
        self.primitive[2]=U[2]/U[0]-1/2*np.power(U[1]/U[0],2)
        self.primitive[2] = np.maximum(self.primitive[2], 1e-8)
    def Pressure(self,gamma=1.4):
        return (gamma-1)*self.primitive[0]*self.primitive[2]
    def F_transform(self):
        U=self.U
        P=self.Pressure()
        self.F[0]=U[1]
        self.F[1]=self.F[0]*self.primitive[1]+P
        self.F[2]=(P+U[2])*self.primitive[1]
    def cs(self):
        return np.sqrt(self.gamma*self.Pressure()/self.primitive[0])
    def eigens(self):
        Cs=self.cs()
        v=self.primitive[1]
        L_plus=v+Cs
        L_minus=v-Cs
        return np.stack((L_plus,L_minus))
    def alpha(self):
        Eigens=self.eigens()
        Alpha=np.empty((2,self.U[0].size-1))
        for i in range(self.n-1):
            Alpha[0,i]=max(Eigens[0,i],Eigens[0,i+1],0)
            Alpha[1,i]=max(-Eigens[1,i],-Eigens[1,i+1],0)
        return Alpha
    def F_HLL(self):
        Alpha = self.alpha()
        F = self.F
        U = self.U

        Flux = np.empty((3, self.n - 1))
        for i in range(self.n - 1):
            t1 = Alpha[0][i] * F[:, i]
            t2 = Alpha[1][i] * F[:, i+1]
            t3 = Alpha[0][i] * Alpha[1][i] * (U[:, i+1] - U[:, i])
            t4 = Alpha[0,i] + Alpha[1,i]
            Flux[:, i] = (t1 + t2 - t3) / t4
        return Flux

    def timeStep(self):
        A = self.alpha()
        dt = 0.5 / np.max(np.maximum(A[0], A[1]))
        Flux = self.F_HLL()
        du= np.zeros(self.U.shape)
        du[:,1:-1] =Flux[:,:-1]-Flux[:,1:]

        
        self.U += du * dt
        self.U_inverse()
        self.F_transform()

        
N = 100
U_init = np.zeros((3, N))
U_init[0, :N//2] = 1.0     # rho
U_init[0, N//2:] = 0.125
U_init[2, :N//2] = 2.5     # E = p/(gamma-1)
U_init[2, N//2:] = 0.25   
test=Hydro(U_init)
plt.plot(test.primitive[0])
for i in range(100):
    test.timeStep()
    
plt.plot(test.primitive[0])
plt.show()