
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


# Kept here (not buried inside the functions) so the solver code stays general.
DOF_PER_NODE       = 2      # Degrees of freedom per node (2D truss: u, v)
EPS_STOP_DEFAULT   = 1e-8   # Newton-Raphson stop tolerance, used if 'eps_stop' is not in the input file
STRESS_TOL         = 1e-9   # Below this |stress| a bar counts as unloaded
PLOT_LINEWIDTH     = 3.5    # Line width for the deformed bars
DISPLACEMENT_SCALE = 1.0    # Magnification applied to displacements when plotting
PRINT_PRECISION    = 3      # Decimal places used when printing numpy arrays

np.set_printoptions(precision=PRINT_PRECISION, suppress=True, linewidth=200)


# ====================================================================================================
#  1. FEA CLASS  -  reads the input file and runs the geometrically non-linear analysis
# ====================================================================================================
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

            # Scalars: integers or floats, e.g. 'nincr = 20;' or 'eps_stop = 1e-8;'
            matches = re.findall(r'(?:\n|^)\s*(\w+)\s*=\s*([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)\s*(?:;|\n)', inp, re.DOTALL)
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
        P = np.zeros((neqn,1))           # Force vector
        stress = np.zeros((ne,1))        # Element stress vector

        # Build the global load vector (this is the final / total load)
        P = buildload(X, IX, ne, P, loads, mprop)

        # ---- Geometrically non-linear truss (Green strain), Newton-Raphson ----
        # 'nincr' (load increments) and 'imax' (max. equilibrium iterations) must be given
        # in the input file; 'eps_stop' is optional and falls back to EPS_STOP_DEFAULT.
        assert hasattr(self, 'nincr'), \
               "Error in input file. Could not read variable nincr (load increments)."
        assert hasattr(self, 'imax'), \
               "Error in input file. Newton-Raphson needs 'imax' (max. equilibrium iterations)."
        nincr    = int(self.nincr)
        imax     = int(self.imax)
        eps_stop = float(getattr(self, 'eps_stop', EPS_STOP_DEFAULT))
        print(f'Non-linear analysis, {nincr} increments, Newton-Raphson (imax = {imax}, eps_stop = {eps_stop:g})')

        D, hist_nr = newton_raphson(X, IX, ne, neqn, mprop, bound, P, nincr, imax, eps_stop, plotdof)
        self.force_disp_nr = hist_nr
        curves = [('Newton-Raphson', hist_nr)]   # (label, history) pairs for the plot

        stress = recover_stress(mprop, X, IX, D, ne, stress)
        self.D = D; self.stress = stress

        # Krenk (3.19) reference curve, only meaningful for the 2-bar Von Mises truss
        ana = analytical_vonmises(mprop, X, IX, hist_nr) if ne == 2 else None
        PlotForceDisplacement(curves, ana)
        PlotStructure(X, IX, ne, neqn, bound, loads, D, stress)  # deformed shape


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


def buildstiff(X, IX, ne, mprop, K, D):
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

        
        B0   = (1.0 / L0**2) * np.array([[-dx], [-dy], [dx], [dy]])
        Amat = np.array([[1,0,-1,0], [0,1,0,-1], [-1,0,1,0], [0,-1,0,1]], dtype=float)

        
        
        edof = np.array([2*n1-2, 2*n1-1, 2*n2-2, 2*n2-1])
        d    = D[edof]                                   

        B_d = (1.0 / L0**2) * (Amat @ d)                 

        eps_G_e   = float(((B0 + 0.5*B_d).T @ d).item()) 
        sigma_G_e = E * eps_G_e                          
        N_G_e     = A * sigma_G_e                        

        # Tangent stiffness contributions (slides "Tangent Stiffness Matrix")
        k_sigma_e = (1.0 / L0**2) * Amat * N_G_e * L0                  
        k_nul_e   = A * E * L0 * (B0 @ B0.T)                           
        k_d_e     = A * E * L0 * (B0 @ B_d.T + B_d @ B0.T + B_d @ B_d.T)

        ke = k_nul_e + k_sigma_e + k_d_e

        K[np.ix_(edof, edof)] += ke  # Add element stiffness to global stiffness matrix
        
    print(f"det her is K {K.toarray()}")
    return K


def internal_force(X, IX, ne, mprop, D, neqn):

    Rint = np.zeros((neqn, 1))
    for e in range(ne):
        n1     = int(IX[e, 0])
        n2     = int(IX[e, 1])
        propno = int(IX[e, 2])

        E = mprop[propno-1, 0]
        A = mprop[propno-1, 1]

        dx = X[n2-1, 0] - X[n1-1, 0]
        dy = X[n2-1, 1] - X[n1-1, 1]
        L0 = np.sqrt(dx**2 + dy**2)

        B0   = (1.0 / L0**2) * np.array([[-dx], [-dy], [dx], [dy]])
        Amat = np.array([[1,0,-1,0], [0,1,0,-1], [-1,0,1,0], [0,-1,0,1]], dtype=float)
        edof = np.array([2*n1-2, 2*n1-1, 2*n2-2, 2*n2-1])
        d    = D[edof]

        B_d   = (1.0 / L0**2) * (Amat @ d)
        eps_G = float(((B0 + 0.5*B_d).T @ d).item())   # Green strain
        N_G_e = A * E * eps_G                          # axial force
 
        Rint[edof] += (B0 + B_d) * N_G_e * L0          # ({B0}+{Bd}) N_G L0

    return Rint


def recover_stress(mprop, X, IX, D, ne, stress):
    # Axial stress in each bar, for the tension/compression colours in PlotStructure.
    # Green strain written with the deformed length L:  eps_G = (L^2 - L0^2) / (2 L0^2)
    for e in range(ne):
        n1     = int(IX[e, 0])
        n2     = int(IX[e, 1])
        propno = int(IX[e, 2])
        E      = mprop[propno-1, 0]

        p1 = X[n1-1] + D[[2*n1-2, 2*n1-1], 0]    # deformed node positions
        p2 = X[n2-1] + D[[2*n2-2, 2*n2-1], 0]

        L0 = np.linalg.norm(X[n2-1] - X[n1-1])   # undeformed length
        L  = np.linalg.norm(p2 - p1)             # deformed length

        stress[e] = E * (L**2 - L0**2) / (2.0 * L0**2)

    return stress


def newton_raphson(X, IX, ne, neqn, mprop, bound, P_final, nincr, imax, eps_stop, plotdof):

    D  = np.zeros((neqn, 1))          # D0 = 0
    P  = np.zeros((neqn, 1))          # accumulated total load, P0 = 0
    dP = P_final / nincr
    tol = eps_stop * np.linalg.norm(P_final)   # absolute stop tolerance

    # Global DOFs with prescribed displacements (0-indexed), for enforcing BC on R
    fixed = [DOF_PER_NODE*int(node) - 2 + int(local_dof) - 1 for node, local_dof, _ in bound]

    pdof   = int(plotdof) - 1
    hist_u = [0.0]
    hist_P = [0.0]

    for n in range(1, nincr + 1):
        P = P + dP                                     # Pn = P^{n-1} + dP
                                                       # D0n = D^{n-1}  (D just carries over)
        for i in range(imax + 1):
            
            R = internal_force(X, IX, ne, mprop, D, neqn) - P   # Ri = Rint(Di) - Pn

           
            R[fixed] = 0.0                             # BC on R: no residual at supports
            res = np.linalg.norm(R)
            if res <= tol:                             # converged -> equilibrium found
                break
            if i == imax:                              # out of iterations, not converged
                print(f'  WARNING: increment {n} did not converge in {imax} iterations (||R|| = {res:.3e})')
                break

            K = sps.csc_matrix((neqn, neqn))
            K = buildstiff(X, IX, ne, mprop, K, D)  # Kt(Di)
            K, rhs = enforce(K, -R, bound)             # BC on Kt and -Ri
            dD = np.linalg.solve(K.toarray(), rhs)     # dDi = -Kt^-1 Ri
            D  = D + dD                                # Di+1 = Di + dDi

        print(f'  NR increment {n:3d}: {i} iterations, ||R|| = {res:.3e}')

        hist_u.append(float(D[pdof].item()))           # Dn = Di
        hist_P.append(float(P[pdof].item()))

    return D, (np.array(hist_u), np.array(hist_P))


# ====================================================================================================
#  6. Plots osv
# ====================================================================================================


def analytical_vonmises(mprop, X, IX, hist):
    # Analytical solution for the symmetric 2-bar Von Mises truss, Krenk (1993), Eq. (3.19):
    #
    #     P = 2 E A (a/L0)^3 [ D/a - 3/2 (D/a)^2 + 1/2 (D/a)^3 ]
    #
    #   a  = undeformed height of the centre node relative to the supports
    #   L0 = undeformed bar length
    #   D  = vertical displacement of the centre node, positive towards the supports
    #
    # The curve is swept far enough to cover the whole snap-through path
    # (D/a = 0 -> 2 is the fully inverted truss). Valid for a << L0.
    E = mprop[0, 0]
    A = mprop[0, 1]

    n1 = int(IX[0, 0]); n2 = int(IX[0, 1])
    dx = X[n2-1, 0] - X[n1-1, 0]
    dy = X[n2-1, 1] - X[n1-1, 1]
    L0 = np.sqrt(dx**2 + dy**2)
    a  = abs(dy)                      # height of the truss

    u_fe  = hist[0]                                    # FE displacements at plotdof
    D_max = max(2.1 * a, 1.05 * max(abs(u_fe)))        # cover FE range and the full curve
    D     = np.linspace(0.0, D_max, 400)

    x = D / a
    P = 2.0 * E * A * (a / L0)**3 * (x - 1.5 * x**2 + 0.5 * x**3)
    return D, P

def PlotForceDisplacement(curves, analytical=None):
    # Force-displacement curves at the plot DOF.
    # `curves` is a list of (label, (u, P)) so several runs can be overlaid.
    plt.figure(2)
    if analytical is not None:
        ua, Pa = analytical
        plt.plot(ua, Pa, 'k-', linewidth=2, label='Analytical')
    for label, (u, P) in curves:
        plt.plot(u, P, 'o--', linewidth=2, markersize=4, label=label)
    plt.xlabel('Displacement at plotdof,  u')
    plt.ylabel('Applied force,  P')
    plt.title('Force-displacement curve')
    plt.grid(True)
    plt.legend()


if __name__ == '__main__':
    # Allow running this file directly (▶ Run) for debugging.
    # Change to the project root so the input file path resolves.
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    Fea('exercise3_1.m')
