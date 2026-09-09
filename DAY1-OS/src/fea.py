
import numpy             as np
import scipy.sparse      as sps
import matplotlib.pyplot as plt
import re
import os, sys

# Make the 'src' package importable whether this file is run via driver.py
# or directly (▶ Run on src/fea.py). Adds the project root to the path.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.plotsupports    import plotsupports
from src.plotloads       import plotloads

# --- Preamble: constants and settings ------------------------------------
# Kept here (not buried inside the functions) so the solver code stays general.
DOF_PER_NODE       = 2      # Degrees of freedom per node (2D truss: u, v)
STRESS_TOL         = 1e-9   # Below this |stress| a bar counts as unloaded
PLOT_LINEWIDTH     = 3.5    # Line width for the deformed bars
DISPLACEMENT_SCALE = 1.0    # Magnification applied to displacements when plotting
PRINT_PRECISION    = 3      # Decimal places used when printing numpy arrays

np.set_printoptions(precision=PRINT_PRECISION, suppress=True, linewidth=200)

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
        neqn = X.shape[0] * DOF_PER_NODE   # Number of equations (DOFs)
        ne = IX.shape[0]                   # Number of elements
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
        
        D = get_displacements(Kmatr, P)                     # Solve for displacements

        strain, stress = recover(mprop, X, IX, D, ne, strain, stress)  # Calculate element stress and strain

        # Plot results
        PlotStructure(X, IX, ne, neqn, bound, loads, D, stress)  # Plot structure






def buildload(X, IX, ne, P, loads, mprop):
    # Assemble the global load vector P from the prescribed nodal loads.
    for i in range(loads.shape[0]):
        node      = int(loads[i, 0])   # node the load is applied to
        local_dof = int(loads[i, 1])   # local DOF at that node (1 = x, 2 = y)
        force     = loads[i, 2]        # load magnitude

        # Map (node, local DOF) to the global DOF index (0-indexed)
        node_dofs  = np.array([DOF_PER_NODE*node-2, DOF_PER_NODE*node-1])
        global_dof = node_dofs[local_dof-1]
        P[global_dof] = force

    return P

def buildstiff(X, IX, ne, mprop, K):
    for e in range(ne):
        # Element nodes and property number (numbered from 1, arrays index from 0)
        n1     = int(IX[e, 0])   # node 1
        n2     = int(IX[e, 1])   # node 2
        propno = int(IX[e, 2])   # property number

        # Material/section properties: mprop row = [ E  A ]
        E = mprop[propno-1, 0]
        A = mprop[propno-1, 1]

        # Element geometry
        dx = X[n2-1, 0] - X[n1-1, 0]
        dy = X[n2-1, 1] - X[n1-1, 1]
        L0 = np.sqrt(dx**2 + dy**2)

        # Strain-displacement vector B0 (4x1)
        B0 = (1.0 / L0**2) * np.array([[-dx], [-dy], [dx], [dy]])

        # Element stiffness matrix: [ke] = A*E*L0 * {B0}{B0}^T
        ke = A * E * L0 * (B0 @ B0.T)

        # Global degrees of freedom for this element (0-indexed):
        # local position i <-> global dof edof[i]
        edof = np.array([2*n1-2, 2*n1-1, 2*n2-2, 2*n2-1])

        K[np.ix_(edof, edof)] += ke  # Add element stiffness to global stiffness matrix

    return K

def enforce(K, P, bound):
    # Apply the prescribed (boundary condition) displacements to K and P.
    constrained_dofs = []

    for i in range(bound.shape[0]):
        node      = int(bound[i, 0])   # constrained node
        local_dof = int(bound[i, 1])   # local DOF at that node (1 = x, 2 = y)
        disp      = bound[i, 2]         # prescribed displacement value

        # Map (node, local DOF) to the global DOF index (0-indexed)
        node_dofs  = np.array([DOF_PER_NODE*node-2, DOF_PER_NODE*node-1])
        global_dof = node_dofs[local_dof-1]

        # Move the known displacement's contribution to the right-hand side
        column = K[:, global_dof].toarray()
        P[global_dof] = disp
        column[global_dof] = 0
        P -= disp * column

        constrained_dofs.append(global_dof)

    # Zero the rows/columns of constrained DOFs and put 1 on the diagonal
    for global_dof in constrained_dofs:
        K[global_dof, :] = 0
        K[:, global_dof] = 0
        K[global_dof, global_dof] = 1

    return K, P

def get_displacements(K, P):
    displacement = np.linalg.solve(K.toarray(), P)
    print(f"This is displacement: {displacement}")
    return displacement

def recover(mprop, X, IX, D, ne, strain, stress):
    for e in range(ne):
        # Element nodes and property number (numbered from 1, arrays index from 0)
        n1     = int(IX[e, 0])   # node 1
        n2     = int(IX[e, 1])   # node 2
        propno = int(IX[e, 2])   # property number

        # Material/section properties: mprop row = [ E  A ]
        E = mprop[propno-1, 0]
        A = mprop[propno-1, 1]

        # Element geometry
        dx = X[n2-1, 0] - X[n1-1, 0]
        dy = X[n2-1, 1] - X[n1-1, 1]
        L0 = np.sqrt(dx**2 + dy**2)

        # Strain-displacement vector B0 (4x1)
        B0 = (1.0 / L0**2) * np.array([[-dx], [-dy], [dx], [dy]])

        edof = np.array([2*n1-2, 2*n1-1, 2*n2-2, 2*n2-1])

        d = D[edof, 0].reshape(-1, 1)  # Element displacement vector (4x1)  

        stress[e] = E * (B0.T @ d)  # Element stress (scalar)
        strain[e] = stress[e] / E  # Element strain (scalar)

    print(f"This is strain: {strain}")
    print(f"This is stress: {stress}")
        
    return strain, stress




def PlotStructure(X, IX, ne, neqn, bound, loads, D, stress):
        # Plot the deformed and undeformed structure

        # Plot settings (values come from the preamble)
        plt.figure(1)
        lw = PLOT_LINEWIDTH        # Line width for plotting bars
        scale = DISPLACEMENT_SCALE # Displacement scaling

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
            # Colour by axial stress: red = compression, blue = tension, green = unloaded
            s = float(stress[e])
            if s < -STRESS_TOL:
                color = 'r'      # compression
            elif s > STRESS_TOL:
                color = 'b'      # tension
            else:
                color = 'g'      # unloaded
            plt.plot(xx_def, yy_def, color, linewidth=lw)

        # Legend describing the stress colours
        from matplotlib.lines import Line2D
        legend_handles = [
            Line2D([0], [0], color='k', linestyle=':', label='Undeformed'),
            Line2D([0], [0], color='b', label='Tension'),
            Line2D([0], [0], color='r', label='Compression'),
            Line2D([0], [0], color='g', label='Non-Loaded'),
            Line2D([0], [0], color='m', label='Force'),
        ]
        plt.legend(handles=legend_handles, loc="upper right")

        # Plot supports and loads
        Xnew, dsup = plotsupports(X, D, neqn, bound)
        plotloads(loads, Xnew, dsup)
        plt.axis('equal')
        plt.show(block=True)


if __name__ == '__main__':
    # Allow running this file directly (▶ Run) for debugging.
    # Change to the project root so the input file path resolves.
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    Fea('competitionv2.m')
    
