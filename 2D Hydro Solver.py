import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class Hydro2D:
    def __init__(self, U_init, gamma=1.4):
        self.gamma = gamma
        # U is conserved variables {rho, rho*vx, rho*vy, E=rho*epsilon+1/2*rho*v^2}
        # condition is {rho, vx, vy, pressure}
        self.U = U_init
        self.Y, self.X = U_init[0].shape[0], U_init[0].shape[1]
        self.condition = np.empty(U_init.shape)
        self.update_conditions()
        
    def U_inverse(self, U):
        # Transforms from U to condition variables
        condition = np.empty_like(U)
        condition[0] = np.maximum(U[0], 1e-8)
        
        # rho = rho
        condition[1] = U[1] / condition[0]  # Fixed: use conditioned rho
        # vx = x_momentum/rho
        condition[2] = U[2] / condition[0]  # Fixed: use conditioned rho
        # pressure = (E - 1/2*(x_momentum*vx + y_momentum*vy))*(gamma-1)
        condition[3] = (U[3] - 1/2*(U[1]*condition[1] + U[2]*condition[2])) * (self.gamma - 1)
        # cap at bottom
        condition[3] = np.maximum(condition[3], 1e-8)
        return condition
    
    def cond_to_U(self, condition):
        # Transforms from cond to conserved
        U = np.empty_like(condition)
        
        U[0] = np.maximum(condition[0], 1e-8)
        # rho = rho
        U[1] = condition[0] * condition[1]
        # x momentum = rho*vx
        U[2] = condition[0] * condition[2]
        # Energy = P/(gamma-1) + 1/2*(x_momentum*vx + y_momentum*vy)
        U[3] = condition[3] / (self.gamma - 1) + 1/2*(U[1]*condition[1] + U[2]*condition[2])
        return U
    
    def update_conditions(self):
        # Updates global condition array
        self.condition = self.U_inverse(self.U)       
    
    def cond_to_Xflux(self, condition):
        # Transforms from condition variables to flux variables
        F = np.empty_like(condition)
        rho = condition[0]
        vx = condition[1]
        vy = condition[2]
        P = condition[3]
        E = P / (self.gamma - 1) + 1/2*rho*(vx**2 + vy**2)
        F[0] = rho * vx
        F[1] = rho * vx**2 + P
        F[2] = rho * vx * vy
        F[3] = (E + P) * vx
        return F
    
    def cond_to_yflux(self, condition):
        # Transforms from condition variables to flux variables
        F = np.empty_like(condition)
        rho = condition[0]
        vx = condition[1]
        vy = condition[2]
        P = condition[3]
        E = P / (self.gamma - 1) + 1/2*rho*(vx**2 + vy**2)
        F[0] = rho * vy
        F[1] = rho * vx * vy 
        F[2] = rho * vy**2 + P
        F[3] = (E + P) * vy
        return F
    
    def cs(self, condition):
        # Speed of sound calculation
        P = condition[3]
        rho = condition[0]
        return np.sqrt(self.gamma * P / rho)
    
    def eigens(self, condition):
        # Eigen calc
        Cs = self.cs(condition)
        v = condition[1]
        L_plus = v + Cs
        L_minus = v - Cs
        return np.stack((L_plus, L_minus))
    
    def minmod(self, a, b, c):
        m = np.stack([a, b, c])
        s = np.sign(m)
        m = np.abs(m)
        return 0.25 * np.abs(s[0] + s[1]) * (s[0] + s[2]) * np.min(m, axis=0)

    def left_interp(self, ci, cl, cr, theta=1.5):
        return ci + 1/2 * self.minmod(
            theta * (ci - cl),
            1/2 * (cr - cl),
            theta * (cr - ci)
        )
    
    def right_interp(self, ci, cr, crX2, theta=1.5):
        return cr - 1/2 * self.minmod(
            theta * (cr - ci),
            1/2 * (crX2 - ci),
            theta * (crX2 - cr)
        )
    
    # Gives interpolated left and right condition variables for a 1-d slice
    def condLR(self, condition):
        n = condition[0].size
        cond = np.pad(condition, ((0, 0), (2, 2)), mode="edge")
        condL = np.empty((4, n - 1))
        condR = np.empty_like(condL)
        condL = self.left_interp(cond[:, 2:-3], cond[:, 1:-4], cond[:, 3:-2])
        # starts from index 2 which is two in front of the added ghosts, then ends two before the added ghosts
        condR = self.right_interp(cond[:, 2:-3], cond[:, 3:-2], cond[:, 4:-1])

        return condL, condR
        
    def alpha(self, condition):
        condL, condR = self.condLR(condition)
        EigensL = self.eigens(condL)
        EigensR = self.eigens(condR)
        Alpha = np.empty((2, condition[0].size - 1))
        Alpha[0, :] = np.maximum(np.maximum(EigensL[0, :], EigensR[0, :]), 0)
        Alpha[1, :] = np.maximum(np.maximum(-EigensL[1, :], -EigensR[1, :]), 0)

        return Alpha, condL, condR
    
    def F_HLL(self, condition, horizontal):
        
        Alpha, condL, condR = self.alpha(condition)
        if horizontal:
            FL, FR = self.cond_to_Xflux(condL), self.cond_to_Xflux(condR)
        else:
            FL, FR = self.cond_to_yflux(condL), self.cond_to_yflux(condR)
        n = condition[0].size
        UL, UR = self.cond_to_U(condL), self.cond_to_U(condR)
        

        max_speed = np.maximum(np.max(Alpha[0]), np.max(Alpha[1]))
        dt = 0.5 / (max_speed + 1e-10)  
        
        Flux = np.empty((4, n - 1))
        t1 = Alpha[0][:] * FL[:, :]
        t2 = Alpha[1][:] * FR[:, :]
        t3 = Alpha[0][:] * Alpha[1][:] * (UR[:, :] - UL[:, :])
        t4 = Alpha[0, :] + Alpha[1, :] + 1e-10  
        Flux[:, :] = (t1 + t2 - t3) / t4

        return Flux, dt


    def L(self, U):
        condition = self.U_inverse(U)
        XFlux = np.empty((4, self.Y, self.X - 1))
        YFlux = np.empty((4, self.Y - 1, self.X))
        dt = 0.1
        
        for i in range(self.Y):
            XFlux[:, i, :], dtnew = self.F_HLL(condition[:, i, :], horizontal=True)
            if dtnew < dt:
                dt = dtnew
        
        for i in range(self.X):
            YFlux[:, :, i], dtnew = self.F_HLL(condition[:, :, i], horizontal=False)
            if dtnew < dt:
                dt = dtnew
        
        du = np.zeros(self.U.shape)
        
        # FIXED: Correct flux differencing for conservation
        du[:, :, :-1] += XFlux  # Left boundary gets positive flux
        du[:, :, 1:] -= XFlux   # Right boundary gets negative flux
        
        du[:, :-1, :] += YFlux  # Bottom boundary gets positive flux
        du[:, 1:, :] -= YFlux   # Top boundary gets negative flux
        
        # FIXED: Return both updated U and the time step
        return U + du * dt, dt
       
    
    def RK3_step(self):
        U0 = self.U.copy()
        
        # Stage 1
        U1, dt = self.L(U0)
        
        # Stage 2
        U2_temp, _ = self.L(U1)
        U2 = 0.75 * U0 + 0.25 * U2_temp
        
        # Stage 3
        U3_temp, _ = self.L(U2)
        U3 = 1/3 * U0 + 2/3 * U3_temp
        
        self.U = U3
        self.update_conditions()
        return dt

    def nSteps(self, n):
        for i in range(n):
            self.RK3_step()


# Setup initial conditions        
N = 100
M = 100
U_init = np.zeros((4, N, M))
l = np.linspace(-1, 1, N)
lx, ly = np.meshgrid(l, l)
start = np.where(lx**2 + ly**2 < 0.01, True, False)
U_init[0] = 1

U_init[3, :, :] = 0.1 / 0.4
U_init[0, start] = 10     # high rho
U_init[3, start] = 1 / 0.4  # high P


# Create simulation
hydro = Hydro2D(U_init)

fig, ax = plt.subplots()
im = ax.imshow(hydro.condition[0], origin='lower', cmap='viridis')
ax.set_title("Density evolution")
plt.colorbar(im, ax=ax)

def update(frame):
    hydro.RK3_step()
    im.set_data(hydro.condition[0])
    im.set_clim(vmin=hydro.condition[0].min(), vmax=hydro.condition[0].max())
    return [im]

anim = FuncAnimation(fig, update, frames=100, interval=30, blit=True)

anim.save("hydro2d_fixed.mp4", writer='ffmpeg', fps=30)