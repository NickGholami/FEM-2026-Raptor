import numpy             as np
import scipy             as sp
import scipy.sparse      as sps
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import math
import re
from scipy.sparse.linalg import spsolve
from scipy.sparse.linalg import splu

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
        Kmatr = sps.csc_matrix((neqn, neqn)) # Stiffness matrix
        P = np.zeros((neqn,1))           # Force vector
        D = np.zeros((neqn,1))           # Displacement vector
        R = np.zeros((neqn,1))           # Residual vector
        R_int = np.zeros((neqn, 1))      # Internal nodal forces
        strain = np.zeros((ne,1))        # Element strain vector
        stress = np.zeros((ne,1))        # Element stress vector
        N = np.zeros((ne, 1))            # Element forces vector


        u_history = [0] # Initialize for plot
        P_history = [0] # Initialize for plot
        nincr = int(self.nincr)
        Pfinal = self.Pfinal
        deltaP = Pfinal / nincr
        load_dof = 2 * (int(loads[0,0]) - 1) + int(loads[0,1])
        deltaP_vec = buildload(X, IX, ne, P.copy(), loads, mprop) * deltaP
        eps_stop = 1e-8

        # Newton Rhapson modified loop
        for i in range(nincr):
            P = P + deltaP_vec
            Kmatr = sps.csc_matrix((neqn, neqn))
            Kmatr = buildstiff(X, IX, ne, mprop, Kmatr, strain)
            Kmatr = enforce_matrix(Kmatr, bound)
            K_factor = splu(Kmatr)

            for iteration in range(int(self.i_max)):
                R_int = np.zeros((neqn, 1))
                strain, stress, N, R_int = recover(mprop, X, IX, D, ne, strain, stress, N, R_int)
                R = R_int - P

                rhs_BC = enforce_rhs(- R, Kmatr, bound)
                if np.linalg.norm(rhs_BC) <= eps_stop * abs(Pfinal):
                    break

                deltaD = K_factor.solve(rhs_BC)
                D = D + deltaD
            else:
                raise RuntimeError(f"Did not converge within max iterations")

            u_history.append(D[int(plotdof) - 1, 0])
            P_history.append(P[load_dof - 1, 0])
        print(D)

        # Plot results
        PlotStructure(X, IX, ne, neqn, bound, loads, D, stress)  # Plot structure
        plot_comparison(X, mprop, u_history, P_history, nincr)



def plot_comparison(X, mprop, u_history, P_history, nincr):
    L = np.linalg.norm(X[-1] - X[0])
    c1, c2, c3, c4, A = mprop[0]
    u_ref = np.linspace(0, u_history[-1], 200)
    P_ref = []

    for u in u_ref:
        eps = u / L
        sigma, _ = material(eps, c1, c2, c3, c4)
        P_ref.append(A * sigma)

    # Sammenlign reference og Euler
    plt.figure(2)
    plt.clf()
    plt.plot(u_ref, P_ref, "black", label="Reference")
    plt.plot(u_history, P_history, "red", label= f"Newton Rhapson modified- {nincr} load increments")
    plt.xlabel("Displacement u")
    plt.ylabel("Force P")
    plt.grid(True)
    plt.legend()
    plt.show()
    



def material(eps, c1, c2, c3, c4):
    lam = 1 + c4 * eps

    if lam <= 0:
        raise ValueError("Material model only works for lam > 0")

    sigma = c1 * (lam - lam**-2) + c2 * (1 - lam**-3) + c3 * (1 - 3 * lam + lam**3 - 2 * lam**-3 + 3 * lam**-2)
    Et = c4 * ( c1 * (1 + 2 * lam**-3) + 3 * c2 * lam**-4 + 3 * c3 * (-1 + lam**2 - 2 * lam**-3 + 2 * lam**-4) )

    return sigma, Et

def buildload(X, IX, ne, P, loads, mprop):
    ndof = 2
    for i in range(loads.shape[0]):
        dof = int((loads[i,0] - 1) * ndof + (loads[i,1] - 1)) 
        P[dof, 0] += loads[i,2]
#        print("Debug stop")
    return P

def buildstiff(X, IX, ne, mprop, K, strain):
    for e in range(ne):
        n1, n2, mat_id = IX[e].astype(int)
        c1, c2, c3, c4, Ae = mprop[mat_id - 1]
        dofs = np.array([n1 * 2 - 2, n1 * 2 - 1, n2 * 2 - 2, n2 * 2 - 1])
        x1 = X[n1 - 1, 0]
        x2 = X[n2 - 1, 0]
        y1 = X[n1 - 1, 1]
        y2 = X[n2 - 1, 1]
        dx = x2 - x1
        dy = y2 - y1
        L0 = np.sqrt(dx**2 + dy**2)
        B0 = 1 / L0**2 * np.array([[-dx, -dy, dx, dy]]).T
        eps = strain[e, 0]
        _, Et = material(eps, c1, c2, c3, c4 )
        ke = Ae * Et * L0 * (B0 @ B0.T)

        # Dobbelt loop:
        for i in range(len(dofs)):
            for j in range(len(dofs)):
                K[dofs[i], dofs[j]] += ke[i,j]
#        print("Debug stop")
    return K

def enforce_matrix(K, bound):
    ndof = 2
    for i in range(bound.shape[0]):
        dof = int((bound[i,0] - 1) * ndof + (bound[i,1] - 1))

        K[dof,:] = 0 # Zero the row
        K[:,dof] = 0 # Zero the column
        K[dof,dof] = 1 # Diagonal to 1
#        print("Debug stop")
    return K

def enforce_rhs(rhs, K, bound):
    ndof = 2
    rhs_BC = rhs.copy()
    for i in range(bound.shape[0]):
        dof = int((bound[i,0] - 1) * ndof + (bound[i,1] - 1))
        u = bound[i,2]
        rhs_BC-= K[:, [dof]].toarray() * u # If displacement is not zero p. 40-41
        rhs_BC[dof,0] = u # Prescribed displacement
    return rhs_BC


def solve_displacements(K,P):
    D = np.linalg.solve(K.toarray(), P)
    return D

def recover(mprop, X, IX, D, ne, strain, stress, N, R_int):
    for e in range(ne):
        n1, n2, mat_id = IX[e].astype(int)
        c1, c2, c3, c4, Ae = mprop[mat_id - 1]
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
        stress[e, 0], _ = material(strain[e,0], c1, c2, c3, c4)
        N[e, 0] = Ae * stress[e, 0]
        R_int[dofs] += B0 * N[e, 0] * L0
#        print('Debug Stop')
    return strain, stress, N, R_int 

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