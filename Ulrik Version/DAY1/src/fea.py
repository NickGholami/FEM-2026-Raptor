# %%

import numpy             as np
import scipy             as sp
import scipy.sparse      as sps
import matplotlib.pyplot as plt
import math
import re
from scipy.sparse.linalg import spsolve

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
        strain = np.zeros((ne,1))        # Element strain vector
        stress = np.zeros((ne,1))        # Element stress vector

        # Calculate displacements
        P = buildload(X, IX, ne, P, loads, mprop)    # Build global load vector

        Kmatr = buildstiff(X, IX, ne, mprop, Kmatr)  # Build global stiffness matrix
        
        Kmatr, P = enforce(Kmatr, P, bound)          # Enforce boundary conditions

        D = getdisplacements(Kmatr, P)               # Solve for displacements
        
        strain, stress = recover(mprop, X, IX, D, ne, strain, stress)  # Calculate element stress and strain

        # Plot results
        PlotStructure(X, IX, ne, neqn, bound, loads, D, stress)  # Plot structure

# %%

def buildload(X, IX, ne, P, loads, mprop):
    for i in range(loads.shape[0]):
        n, d, pe = loads[i]
        P[int(2 * n - 2 + d - 1)] = pe
    return P

def buildstiff(X, IX, ne, mprop, K):
    for e in range(ne):
        n1, n2, mat_id = IX[ e ].astype( int )
        xe = np.array([ X[n1 - 1, 0], X[n1 - 1, 1], X[n2 - 1, 0], X[n2 - 1, 1]])
        dx = xe[2] - xe[0]
        dy = xe[3] - xe[1]
        L0 = math.sqrt(dx**2 + dy**2)
        Ee = mprop[mat_id - 1, 0]
        Ae = mprop[mat_id - 1, 1]
        B0 = 1 / (L0**2) * np.array([[-dx, -dy, dx, dy]]).T
        ke = Ee * Ae * L0 * B0 @ B0.T
        edofT = np.array([n1 * 2 -1, n1 * 2, n2 * 2 -1, n2 * 2])
        # print("edofT:", edofT)
        # print("ke:", ke)
        K[ np.ix_( edofT - 1, edofT - 1 ) ] += ke
    #print(K.toarray())        
    return K

def enforce(K, P, bound):

    for i in range(bound.shape[0]):

        n, dof, disp = bound[i]

        idof = int(2 * n - 2 + dof)

        P[:, 0] -= K[:, idof - 1].toarray().flatten() * disp

        K[idof - 1, :] = 0
        K[:, idof - 1] = 0
        K[idof - 1, idof - 1] = 1

        P[idof - 1, 0] = disp
    return K, P

def getdisplacements(K, P):
    D = np.linalg.solve(K.toarray(), P)
    print(D)
    return D

def recover(mprop, X, IX, D, ne, strain, stress):
    for e in range(ne):
        n1, n2, mat_id = IX[e].astype(int)
        xe = np.array([ X[n1 - 1, 0], X[n1 - 1, 1], X[n2 - 1, 0], X[n2 - 1, 1]])
        dx = xe[2] - xe[0]
        dy = xe[3] - xe[1]
        L0 = math.sqrt(dx**2 + dy**2)
        Ee = mprop[mat_id - 1, 0]
        Ae = mprop[mat_id - 1, 1]
        B0 = 1 / (L0**2) * np.array([[-dx, -dy, dx, dy]]).T
        de = D[np.array([2*n1 - 2, 2*n1 - 1, 2*n2 - 2, 2*n2 - 1]), 0]
        stress[e] = Ee * B0.T @ de
        strain[e] = B0.T @ de
    print(f'strain: {strain}, stress: {stress}')
    return strain, stress

def PlotStructure(X, IX, ne, neqn, bound, loads, D, stress):
    from matplotlib.lines import Line2D

    # Plot settings
    plt.figure(1)
    plt.clf()
    lw = 3.5
    scale = 1.0

    # Tolerance for deciding whether a bar is unloaded
    max_stress = np.max(np.abs(stress))
    if max_stress > 0:
        stress_tol = 1e-6 * max_stress
    else:
        stress_tol = 1e-12

    # Plot elements
    for e in range(ne):
        xx = X[IX[e, 0:2].astype(int) - 1, 0]
        yy = X[IX[e, 0:2].astype(int) - 1, 1]
        plt.plot(xx, yy, "k:", linewidth=1)

        # Get element displacement
        n1, n2 = IX[e, 0:2].astype(int)
        edof = np.array([2*n1, 2*n1 + 1, 2*n2, 2*n2 + 1])
        xx_def = xx + scale * D[edof[0:4:2] - 2, 0]
        yy_def = yy + scale * D[edof[1:4:2] - 2, 0]

        # Choose color based on stress
        sigma = float(stress[e, 0])
        if sigma > stress_tol:
            color = "blue"       # Tension
        elif sigma < -stress_tol:
            color = "red"        # Compression
        else:
            color = "green"      # Essentially zero stress
        plt.plot(xx_def, yy_def, color=color, linewidth=lw)

    # Legend
    legend_elements = [
        Line2D([0], [0], color="black", linestyle=":", linewidth=1, label="Undeformed"),
        Line2D([0], [0], color="blue", linewidth=lw, label="Tension"),
        Line2D([0], [0], color="red", linewidth=lw, label="Compression"),
        Line2D([0], [0], color="green", linewidth=lw, label="No tension/compression")
    ]
    plt.legend(handles=legend_elements, loc="upper right")

    # Plot supports and loads
    Xnew, dsup = plotsupports(X, D, neqn, bound)
    plotloads(loads, Xnew, dsup)
    plt.axis("equal")
    plt.show(block=True)



# def PlotStructure(X, IX, ne, neqn, bound, loads, D, stress):
#         # Plot the deformed and undeformed structure

#         # Plot settings
#         plt.figure(1)
#         lw = 3.5        # Linewidth for plotting bars
#         scale = 1.0     # Displacement scaling

#         for e in range(ne):
#             xx = X[IX[e, 0:2].astype(int)-1, 0]
#             yy = X[IX[e, 0:2].astype(int)-1, 1]
#             # Plot undeformed solution
#             plt.plot(xx, yy, 'k:', linewidth=1)
#             # Get displacements in x and y
#             n1, n2 = IX[e, 0:2].astype(int)
#             edof = np.array([2*n1, 2*n1 + 1, 2*n2, 2*n2 + 1])
#             xx_def = xx + scale*D[edof[0:4:2]-2,0]
#             yy_def = yy + scale*D[edof[1:4:2]-2,0]
#             plt.plot(xx_def, yy_def, 'b', linewidth=lw)

#         plt.legend(["Undeformed", "Deformed"], loc="upper right")

#         # Plot supports and loads 
#         Xnew, dsup = plotsupports(X, D, neqn, bound)
#         plotloads(loads, Xnew, dsup)
#         plt.axis('equal')
#         plt.show(block=True)
# %%
