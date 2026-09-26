import numpy             as np
import scipy             as sp
import scipy.sparse      as sps
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import re

from src.plotsupports    import plotsupports
from src.plotloads       import plotloads

class Fea:
    def __init__(self, input_file):       
        # Read input in the standard Matlab format and convert to python variables
        with open(input_file, 'r') as file:
            inp = file.read()

            matches = re.findall(r'(?:\n|^)\s*(\w+)\s*=\s*\[\s*([\s\S]*?)\s*\]', inp, re.DOTALL)
            for name,content in matches:
                vals = content.strip().split('\n')
                vals = np.array([[float(val) for val in row.strip().split()] for row in vals])
                setattr(self, name, vals)     # this is for storing a variable in a class

            matches = re.findall(r'(?:\n|^)\s*(\w+)\s*=\s*(\d+)\s*(?:;|\n)', inp, re.DOTALL)
            for name,content in matches:
                setattr(self, name, float(content.strip()))

            for requiredvar in ("X", "IX", "mprop", "bound", "loads", "plotdof"):
                assert hasattr(self, requiredvar), \
                       f"Error in input file. Could not read variable {requiredvar}."
            X = self.X; IX = self.IX; mprop = self.mprop
            bound = self.bound; loads = self.loads; plotdof = self.plotdof


        # Calculate problem size
        neqn = X.shape[0] * X.shape[1]   # Number of equations
        ne = IX.shape[0]                 # Number of elements
        print(f'Number of DOF {neqn} Number of elements {ne}')

        # Initialize arrays
        Kmatr = sps.csc_matrix((neqn, neqn))    # Stiffness matrix
        P = np.zeros((neqn,1))                  # Force vector
        D = np.zeros((neqn,1))                  # Displacement vector
        R = np.zeros((neqn,1))                  # Residual vector
        R_int = np.zeros((neqn, 1))             # Internal nodal forces
        strain = np.zeros((ne,1))               # Element strain vector
        stress = np.zeros((ne,1))               # Element stress vector
        N = np.zeros((ne, 1))                   # Element forces vector
    



        


        # Topology Loop:
        max_iopt = 200
        V = np.zeros((ne, 1))
        V_initial, V = compute_volume(X, IX, ne, mprop, V)
        V_s = 0.2 * V_initial
        print(f'Initial Volume: \n {V_initial}')
        rho = np.ones((ne, 1)) * V_s / V_initial
        print(f'Initial rho: \n {rho}')
        epsilon = 1e-6
        f_history = []
        P_ext = buildload(X, IX, ne, np.zeros((neqn, 1)), loads, mprop)
        p = 2 # Penalization exponent
        for iteration in range(max_iopt):
            rho_old = rho
            Kmatr = sps.csc_matrix((neqn, neqn))
            Kmatr = buildstiff(X, IX, ne, mprop, Kmatr, rho, p)     # Build global stiffness matrix
            Kmatr, P = enforce(X, Kmatr, P_ext.copy(), bound)          # Enforce boundary conditions
            D = solve_displacements(Kmatr, P)                       # Solve for displacments
            R_int = np.zeros((neqn, 1))
            strain, stress, N, R_int, nabla_f, SED = recover(mprop, X, IX, D, ne, strain, stress, N, R_int, rho, p, V)
            f_history.append((D.T @ P).item())
            nabla_g = V

            rho = bisect(rho_old, V_s, nabla_f, nabla_g)
            if np.linalg.norm(rho_old - rho) <= epsilon * np.linalg.norm(rho):
                break
            print(f'Iteration Number: {iteration+1}')

        plt.figure(2)
        plt.clf()
        plt.plot(range(1, len(f_history) + 1), f_history)
        plt.xlabel("Optimization iteration")
        plt.ylabel("Compliance")
        plt.grid(True)

        # Plot results
        PlotStructure(X, IX, ne, neqn, bound, loads, D, stress)  # Plot structure
        PlotStructure_topo(X, IX, ne, neqn, bound, loads, D, stress, rho) # Plot topo scaled structure
        print(f'Displacements: \n {D}')
        print(f'rho: \n {rho}')
        print(f'Strain Energy Density: \n {SED}')

# %%

def bisect(rho_old, V_s, nabla_f, nabla_g):
    lam1 = 1e-10
    lam2 = 1e10
    epsilon = 1e-5
    rho_min = 1e-6
    eta = 0.5
    while (lam2 - lam1) / (lam1 + lam2) > epsilon:
        lam_mid = (lam1 + lam2) / 2
        B = (-nabla_f/ (lam_mid * nabla_g))
        rho = rho_old * B**eta
        rho = np.clip(rho, rho_min, 1.0)
        v = nabla_g
        g = (rho.T @ v).item() - V_s
        if g > 0:
            lam1 = lam_mid
        else:
            lam2 = lam_mid
    return rho


def compute_volume(X, IX, ne, mprop, V):
    for e in range(ne):
        n1, n2, mat_id = IX[e].astype(int)
        Ae = mprop[mat_id - 1, 1]
        x1 = X[n1 - 1, 0]
        x2 = X[n2 - 1, 0]
        y1 = X[n1 - 1, 1]
        y2 = X[n2 - 1, 1]
        dx = x2 - x1
        dy = y2 - y1
        L0 = np.sqrt(dx**2 + dy**2)
        V[e,0] = L0 * Ae
    V_initial = np.sum(V)
    return V_initial, V

def buildload(X, P, loads):
    ndof = X.shape[1]
    for i in range(loads.shape[0]):
        dof = int((loads[i,0] - 1) * ndof + (loads[i,1] - 1)) 
        P[dof, 0] += loads[i,2]
        # print("Debug stop")
    return P

def buildstiff(X, IX, ne, mprop, K, rho, p):
    for e in range(ne):
        n1, n2, mat_id = IX[e].astype(int)
        Ee = mprop[mat_id - 1, 0]
        Ae = mprop[mat_id - 1, 1]
        dofs = np.array([n1 * 2 - 2, n1 * 2 - 1, n2 * 2 - 2, n2 * 2 - 1])
        x1 = X[n1 - 1, 0]
        x2 = X[n2 - 1, 0]
        y1 = X[n1 - 1, 1]
        y2 = X[n2 - 1, 1]
        dx = x2 - x1
        dy = y2 - y1
        L0 = np.sqrt(dx**2 + dy**2)
        B0 = 1 / L0**2 * np.array([[-dx, -dy, dx, dy]]).T
        k0 = Ae * Ee * L0 * (B0 @ B0.T)
        ke = rho[e,0]**p * k0

        # Double loop:
        for i in range(len(dofs)):
            for j in range(len(dofs)):
                K[dofs[i], dofs[j]] += ke[i,j]
    return K

def enforce(X, K, P, bound):
    ndof = X.shape[1]
    for i in range(bound.shape[0]):
        dof = int((bound[i,0] - 1) * ndof + (bound[i,1] - 1))

        u = bound[i,2]
        P-= K[:, [dof]].toarray() * u # If displacement is not zero p. 40-41
        K[dof,:] = 0 # Zero the row
        K[:,dof] = 0 # Zero the column
        K[dof,dof] = 1 # Diagonal to 1
        P[dof,0] = u # Prescribed displacement
#        print("Debug stop")
    return K, P

def solve_displacements(K,P):
    D = np.linalg.solve(K.toarray(), P)
    return D

def recover(mprop, X, IX, D, ne, strain, stress, N, R_int, rho, p, V):
    napla_f = np.zeros((ne, 1))
    SED = np.zeros((ne, 1))
    for e in range(ne):
        n1, n2, mat_id = IX[e].astype(int)
        Ee = mprop[mat_id - 1, 0]
        Ae = mprop[mat_id - 1, 1]
        dofs = np.array([n1 * 2 - 2, n1 * 2 - 1, n2 * 2 - 2, n2 * 2 - 1])
        x1 = X[n1 - 1, 0]
        x2 = X[n2 - 1, 0]
        y1 = X[n1 - 1, 1]
        y2 = X[n2 - 1, 1]
        dx = x2 - x1
        dy = y2 - y1
        L0 = np.sqrt(dx**2 + dy**2)
        B0 = 1 / L0**2 * np.array([[-dx, -dy, dx, dy]]).T
        d = D[dofs]
        strain[e,0] = (d.T @ B0).item()
        stress[e, 0] = Ee * strain[e,0]
        N[e, 0] = Ae * stress[e, 0]
        R_int[dofs] += B0 * N[e, 0] * L0
        k0 = Ae * Ee * L0 * (B0 @ B0.T)
        napla_f[e,0] = -p * rho[e,0]**(p-1) * d.T @ k0 @ d
        ke = rho[e,0]**p * k0
        SED[e,0] = (d.T @ ke @ d) / (V[e,0] * rho[e,0])
#        print('Debug Stop')
    return strain, stress, N, R_int, napla_f, SED

def PlotStructure(X, IX, ne, neqn, bound, loads, D, stress):
        # Plot the deformed and undeformed structure

        # Plot settings
        plt.figure(1)
        plt.clf()
        lw = 3.5        # Linewidth for plotting bars
        scale = 1.0     # Displacement scaling
        tol = 1e-5
        for e in range(ne):
            xx = X[IX[e, 0:2].astype(int)-1, 0]
            yy = X[IX[e, 0:2].astype(int)-1, 1]
            # Plot undeformed solution
            plt.plot(xx, yy, 'k:', linewidth=1)
            # Get displacements in x and y
            n1, n2 = IX[e, 0:2].astype(int)
            edof = np.array([2*n1, 2*n1 + 1, 2*n2, 2*n2 + 1])
            xx_def = xx + scale*D[edof[0:4:2]-2,0]
            yy_def = yy + scale*D[edof[1:4:2]-2,0]
            sigma = stress[e,0]
            if sigma > tol:
                colour = 'blue'
            elif sigma < - tol:
                colour = 'red'
            else:
                colour = 'green'
            plt.plot(xx_def, yy_def, colour, linewidth=lw)

        legend_lines = [
            Line2D([0], [0], color = 'black', linestyle = ':', label = 'undeformed'),
            Line2D([0], [0], color = 'blue', linestyle= '-', label = 'tension'),
            Line2D([0], [0], color = 'red', linestyle= '-', label = 'compression'),
            Line2D([0], [0], color = 'green', linestyle= '-', label = 'unloaded'),
        ]

        plt.legend(handles = legend_lines, loc="upper right")

        # Plot supports and loads 
        Xnew, dsup = plotsupports(X, D, neqn, bound)
        plotloads(loads, Xnew, dsup)
        plt.axis('equal')
        plt.show(block=True)

def PlotStructure_topo(X, IX, ne, neqn, bound, loads, D, stress, rho):
        # Plot the deformed and undeformed structure

        # Plot settings
        plt.figure(1)
        plt.clf()
        scale = 1.0     # Displacement scaling
        tol = 1e-5
        for e in range(ne):
            density = float(rho[e,0])

            if density < 0.05:
                continue  # Skjuler kun stangen i plottet

            xx = X[IX[e, 0:2].astype(int)-1, 0]
            yy = X[IX[e, 0:2].astype(int)-1, 1]

            # Get displacements in x and y
            n1, n2 = IX[e, 0:2].astype(int)
            edof = np.array([2*n1, 2*n1 + 1, 2*n2, 2*n2 + 1])
            xx_def = xx + scale*D[edof[0:4:2]-2,0]
            yy_def = yy + scale*D[edof[1:4:2]-2,0]
            sigma = stress[e,0]
            if sigma > tol:
                colour = 'blue'
            elif sigma < - tol:
                colour = 'red'
            else:
                colour = 'green'
            plt.plot(xx_def, yy_def, colour, linewidth=density)

        legend_lines = [
            Line2D([0], [0], color = 'blue', linestyle= '-', label = 'tension'),
            Line2D([0], [0], color = 'red', linestyle= '-', label = 'compression'),
            Line2D([0], [0], color = 'green', linestyle= '-', label = 'unloaded'),
        ]

        plt.legend(handles = legend_lines, loc="upper right")

        # Plot supports and loads 
        Xnew, dsup = plotsupports(X, scale * D, neqn, bound)
        plotloads(loads, Xnew, dsup)
        plt.axis('equal')
        plt.show(block=True)

