
import numpy             as np
import scipy.sparse      as sps
import scipy.linalg      as spl
import matplotlib.pyplot as plt
import re
import os, sys

# Make the 'src' package importable whether this file is run via driver.py
# or directly (▶ Run on src/fea.py). Adds the project root to the path.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.plotsupports    import plotsupports
from src.plotloads       import plotloads


# --- File layout ---------------------------------------------------------
#   1. Fea class               : reads the input file and runs Day 1 or Day 2
#   2. SHARED                  : used by both Day 1 and Day 2
#   3. DAY 1                   : linear material, direct solve
#   4. DAY 2 (shared)          : non-linear rubber material helpers
#   5. DAY 2a                  : pure Euler method
#   6. DAY 2b                  : Euler with one-step equilibrium-correction
#   7. DAY 2c                  : Newton-Raphson equilibrium iterations
#   8. DAY 2d                  : modified Newton-Raphson (Kt factorized once per load step)
#   9. DAY 2 post-processing   : analytical reference and force-displacement plot

# --- Preamble: constants and settings ------------------------------------
# Kept here (not buried inside the functions) so the solver code stays general.
DOF_PER_NODE       = 2      # Degrees of freedom per node (2D truss: u, v)
EPS_STOP_DEFAULT   = 1e-8   # Newton-Raphson stop tolerance, used if 'eps_stop' is not in the input file
STRESS_TOL         = 1e-9   # Below this |stress| a bar counts as unloaded
PLOT_LINEWIDTH     = 3.5    # Line width for the deformed bars
DISPLACEMENT_SCALE = 1.0    # Magnification applied to displacements when plotting
PRINT_PRECISION    = 3      # Decimal places used when printing numpy arrays

np.set_printoptions(precision=PRINT_PRECISION, suppress=True, linewidth=200)


# ====================================================================================================
#  1. FEA CLASS  -  reads input, picks Day 1 (linear) or Day 2 (non-linear, if nincr is given)
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
        Kmatr = sps.csc_matrix((neqn, neqn)) # Stiffness matrix
        P = np.zeros((neqn,1))           # Force vector
        D = np.zeros((neqn,1))           # Displacement vector
        R = np.zeros((neqn,1))           # Residual vector
        strain = np.zeros((ne,1))        # Element strain vector
        stress = np.zeros((ne,1))        # Element stress vector

        # Build the global load vector (this is the final / total load)
        P = buildload(X, IX, ne, P, loads, mprop)

        if hasattr(self, 'nincr'):
            # ---- DAY 2: non-linear rubber material, incremental methods ----
            # Triggered automatically when the input file defines 'nincr'.
            # Choose the solver with the input variable 'method':
            #     method = 1  ->  pure Euler
            #     method = 2  ->  Euler + one-step equilibrium-correction (Ex. 2.2)
            #     method = 3  ->  Newton-Raphson equilibrium iterations (Ex. 2.3)
            #     method = 4  ->  modified Newton-Raphson (Ex. 2.4)
            #     (omitted)   ->  run ALL of them and overlay them for comparison
            nincr  = int(self.nincr)
            method = int(getattr(self, 'method', 0))
            names  = {0: 'all', 1: 'pure Euler', 2: 'Euler + equilibrium-correction',
                      3: 'Newton-Raphson', 4: 'modified Newton-Raphson'}
            print(f'Non-linear analysis, {nincr} increments, method = {method} ({names.get(method, "?")})')

            curves = []          # (label, history) pairs to overlay in the plot
            D = None             # displacement used for stress recovery / structure plot

            if method in (0, 1):
                D_e, hist_e = euler_solve(X, IX, ne, neqn, mprop, bound, P, nincr, plotdof)
                curves.append(('Euler', hist_e))
                self.force_disp = hist_e
                D = D_e

            if method in (0, 2):
                D_c, hist_c = Euler_equilibrium_correction(X, IX, ne, neqn, mprop, bound, P, nincr, plotdof)
                curves.append(('Euler + correction', hist_c))
                self.force_disp_corr = hist_c
                D = D_c          # prefer the corrected displacement when available

            if method in (0, 3, 4):
                # Both Newton-Raphson variants need the equilibrium-iteration settings
                assert hasattr(self, 'imax'), \
                       "Error in input file. Newton-Raphson needs 'imax' (max. equilibrium iterations)."
                imax     = int(self.imax)
                eps_stop = float(getattr(self, 'eps_stop', EPS_STOP_DEFAULT))

            if method in (0, 3):
                D_nr, hist_nr = newton_raphson(X, IX, ne, neqn, mprop, bound, P, nincr, imax, eps_stop, plotdof)
                curves.append(('Newton-Raphson', hist_nr))
                self.force_disp_nr = hist_nr
                D = D_nr         # NR is in equilibrium, so prefer it for the structure plot

            if method in (0, 4):
                D_mnr, hist_mnr = modified_newton_raphson(X, IX, ne, neqn, mprop, bound, P, nincr, imax, eps_stop, plotdof)
                curves.append(('Modified NR', hist_mnr))
                self.force_disp_mnr = hist_mnr
                D = D_mnr        # also converged, so equally good for the structure plot

            strain, stress = recover_nonlinear(mprop, X, IX, D, ne, strain, stress)
            self.D = D; self.stress = stress

            ana = analytical_curve(mprop, X, IX, ne, curves[-1][1])  # reference curve
            PlotForceDisplacement(curves, ana)
            PlotStructure(X, IX, ne, neqn, bound, loads, D, stress)  # deformed shape

        else:
            # ---- DAY 1: linear material, direct solve ----
            Kmatr = buildstiff(X, IX, ne, mprop, Kmatr)  # Build global stiffness matrix
            Kmatr, P = enforce(Kmatr, P, bound)          # Enforce boundary conditions
            D = get_displacements(Kmatr, P)              # Solve for displacements
            strain, stress = recover(mprop, X, IX, D, ne, strain, stress)  # stress & strain

            self.D = D; self.stress = stress

            PlotStructure(X, IX, ne, neqn, bound, loads, D, stress)  # Plot structure



# ====================================================================================================
#  2. SHARED  -  used by both Day 1 and Day 2
# ====================================================================================================
#  buildload, enforce, PlotStructure
# ====================================================================================================

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



# ====================================================================================================
#  3. DAY 1  -  linear material, direct solve  (K D = P)
# ====================================================================================================
#  mprop row = [ E  A ]
#  buildstiff, get_displacements, recover
# ====================================================================================================

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



# ====================================================================================================
#  4. DAY 2 (shared)  -  non-linear rubber material, used by ALL Day 2 solvers
# ====================================================================================================
#  mprop row = [ A  c1  c2  c3  c4 ]
#  The elastic modulus E is replaced by the tangent modulus Et(eps).
#  sigma, tangent_modulus, build_tangent, recover_nonlinear
# ====================================================================================================

def sigma(eps, c1, c2, c3, c4):
    # Signorini stress-strain relation, Eq. (2.21). lam = stretch.
    lam = 1.0 + c4 * eps
    return ( c1 * (lam - lam**-2)
           + c2 * (1.0 - lam**-3)
           + c3 * (1.0 - 3.0*lam + lam**3 - 2.0*lam**-3 + 3.0*lam**-2) )

def tangent_modulus(eps, c1, c2, c3, c4):
    # Tangent stiffness modulus  Et = dsigma/deps, Eq. (2.22).
    lam = 1.0 + c4 * eps
    return c4 * ( c1 * (1.0 + 2.0*lam**-3)
                + 3.0 * c2 * lam**-4
                + 3.0 * c3 * (-1.0 + lam**2 - 2.0*lam**-3 + 2.0*lam**-4) )

def build_tangent(X, IX, ne, mprop, D, K):
    # Global TANGENT stiffness matrix evaluated at the current displacement D.
    # Same shape as buildstiff, but E is replaced by Et(eps) per element.
    for e in range(ne):
        n1     = int(IX[e, 0])
        n2     = int(IX[e, 1])
        propno = int(IX[e, 2])

        A, c1, c2, c3, c4 = mprop[propno-1]   # rubber property row

        dx = X[n2-1, 0] - X[n1-1, 0]
        dy = X[n2-1, 1] - X[n1-1, 1]
        L0 = np.sqrt(dx**2 + dy**2)

        B0   = (1.0 / L0**2) * np.array([[-dx], [-dy], [dx], [dy]])
        edof = np.array([2*n1-2, 2*n1-1, 2*n2-2, 2*n2-1])

        eps = float((B0.T @ D[edof]).item())  # current element strain
        Et  = tangent_modulus(eps, c1, c2, c3, c4)

        kt = Et * A * L0 * (B0 @ B0.T)        # local tangent stiffness
        K[np.ix_(edof, edof)] += kt

    return K

def recover_nonlinear(mprop, X, IX, D, ne, strain, stress):
    # Element strain and (rubber) stress from the final displacement D.
    for e in range(ne):
        n1     = int(IX[e, 0])
        n2     = int(IX[e, 1])
        propno = int(IX[e, 2])

        A, c1, c2, c3, c4 = mprop[propno-1]

        dx = X[n2-1, 0] - X[n1-1, 0]
        dy = X[n2-1, 1] - X[n1-1, 1]
        L0 = np.sqrt(dx**2 + dy**2)

        B0   = (1.0 / L0**2) * np.array([[-dx], [-dy], [dx], [dy]])
        edof = np.array([2*n1-2, 2*n1-1, 2*n2-2, 2*n2-1])

        eps = float((B0.T @ D[edof]).item())
        strain[e] = eps
        stress[e] = sigma(eps, c1, c2, c3, c4)

    print(f"This is strain: {strain}")
    print(f"This is stress: {stress}")
    return strain, stress



# ====================================================================================================
#  5. DAY 2a  -  pure Euler method
# ====================================================================================================
#  euler_solve  (uses build_tangent, enforce)
# ====================================================================================================

def euler_solve(X, IX, ne, neqn, mprop, bound, P_final, nincr, plotdof):
    # Forward-Euler incremental solution (slides "Pseudo-code (Euler Method)").
    #   P0 = 0, D0 = 0, dP = P_final / nincr
    #   for n = 1..nincr:  Kt(D^{n-1}) dD = dP ;  D^n = D^{n-1} + dD
    D  = np.zeros((neqn, 1))          # D0 = 0
    P  = np.zeros((neqn, 1))          # accumulated total load, P0 = 0
    dP = P_final / nincr

    pdof   = int(plotdof) - 1          # input DOF is 1-indexed
    hist_u = [0.0]                     # displacement history at plotdof
    hist_P = [0.0]                     # applied-force history at plotdof

    for n in range(1, nincr + 1):
        P = P + dP                                     # Pn = P^{n-1} + dP
        K = sps.csc_matrix((neqn, neqn))
        K = build_tangent(X, IX, ne, mprop, D, K)      # tangent from D^{n-1}
        K, rhs = enforce(K, dP.copy(), bound)          # BC on Kt and dP
        # rhs er basically bare Delta P
        dD = np.linalg.solve(K.toarray(), rhs)         # solve Kt dD = dP
        D  = D + dD                                    # accumulate

        hist_u.append(float(D[pdof].item()))
        hist_P.append(float(P[pdof].item()))

    return D, (np.array(hist_u), np.array(hist_P))



# ====================================================================================================
#  6. DAY 2b  -  Euler method with one-step equilibrium-correction  (Exercise 2.2)
# ====================================================================================================
#  internal_force, Euler_equilibrium_correction  (also uses build_tangent, enforce)
#  NB: internal_force is also used by Newton-Raphson (section 7)
# ====================================================================================================

def internal_force(X, IX, ne, mprop, D, neqn):
    # Global INTERNAL force vector at displacement D:  Rint = sum_e {B0} N^e L0^e
    # with element axial force  N^e = A * sigma(eps)   (Eq. 2.11 / 2.12).
    # Same element loop as build_tangent, but it assembles a FORCE VECTOR using
    # the true stress sigma(eps) instead of a stiffness matrix using Et(eps).
    Rint = np.zeros((neqn, 1))
    for e in range(ne):
        n1     = int(IX[e, 0])
        n2     = int(IX[e, 1])
        propno = int(IX[e, 2])

        A, c1, c2, c3, c4 = mprop[propno-1]

        dx = X[n2-1, 0] - X[n1-1, 0]
        dy = X[n2-1, 1] - X[n1-1, 1]
        L0 = np.sqrt(dx**2 + dy**2)

        B0   = (1.0 / L0**2) * np.array([[-dx], [-dy], [dx], [dy]])
        edof = np.array([2*n1-2, 2*n1-1, 2*n2-2, 2*n2-1])

        eps = float((B0.T @ D[edof]).item())        # current element strain
        Ne  = A * sigma(eps, c1, c2, c3, c4)         # element axial force N^e

        Rint[edof] += B0 * Ne * L0                   # assemble {B0} N^e L0^e

    return Rint

def Euler_equilibrium_correction(X, IX, ne, neqn, mprop, bound, P_final, nincr, plotdof):
    # Euler method WITH one-step equilibrium-correction (slides / Exercise 2.2).
    #   P0 = 0, D0 = 0, R0 = 0, dP = P_final / nincr
    #   for n = 1..nincr:
    #       Pn  = P^{n-1} + dP                         # accumulated total load
    #       Kt(D^{n-1})
    #       enforce BC on Kt and (dP - R^{n-1})
    #       dD  = Kt^-1 (dP - R^{n-1})                 # <-- correction: subtract residual
    #       Dn  = D^{n-1} + dD
    #       Rn  = Rint(Dn) - Pn                        # <-- measure new residual
    D  = np.zeros((neqn, 1))          # D0 = 0
    R  = np.zeros((neqn, 1))          # R0 = 0  (residual from previous step)
    P  = np.zeros((neqn, 1))          # accumulated total load, starts at 0
    dP = P_final / nincr

    pdof   = int(plotdof) - 1
    hist_u = [0.0]
    hist_P = [0.0]

    for n in range(1, nincr + 1):
        P = P + dP                                     # Pn = P^{n-1} + dP
        K = sps.csc_matrix((neqn, neqn))
        K = build_tangent(X, IX, ne, mprop, D, K)      # Kt(D^{n-1})
        K, rhs = enforce(K, dP - R, bound)             # BC on Kt and (dP - R^{n-1})
        dD = np.linalg.solve(K.toarray(), rhs)         # dD = Kt^-1 (dP - R^{n-1})
        D  = D + dD                                    # Dn = D^{n-1} + dD

        R  = internal_force(X, IX, ne, mprop, D, neqn) - P   # Rn = Rint(Dn) - Pn

        hist_u.append(float(D[pdof].item()))
        hist_P.append(float(P[pdof].item()))

    return D, (np.array(hist_u), np.array(hist_P))



# ====================================================================================================
#  7. DAY 2c  -  Newton-Raphson equilibrium iterations  (Exercise 2.3)
# ====================================================================================================
#  newton_raphson  (uses internal_force, build_tangent, enforce)
# ====================================================================================================

def newton_raphson(X, IX, ne, neqn, mprop, bound, P_final, nincr, imax, eps_stop, plotdof):
    # Incremental scheme with Newton-Raphson equilibrium iterations (slides "Newton-Raphson algorithm").
    #   for n = 1..nincr:
    #       Pn   = P^{n-1} + dP
    #       D0n  = D^{n-1}
    #       for i = 0..imax:
    #           Ri   = Rint(Di) - Pn
    #           enforce BC on Ri
    #           stop when ||Ri|| <= eps_stop * ||P_final||
    #           Kt(Di), enforce BC on Kt
    #           dDi  = -Kt^-1 Ri
    #           Di+1 = Di + dDi
    #       Dn = Di
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
            K = build_tangent(X, IX, ne, mprop, D, K)  # Kt(Di)
            K, rhs = enforce(K, -R, bound)             # BC on Kt and -Ri
            dD = np.linalg.solve(K.toarray(), rhs)     # dDi = -Kt^-1 Ri
            D  = D + dD                                # Di+1 = Di + dDi

        print(f'  NR increment {n:3d}: {i} iterations, ||R|| = {res:.3e}')

        hist_u.append(float(D[pdof].item()))           # Dn = Di
        hist_P.append(float(P[pdof].item()))

    return D, (np.array(hist_u), np.array(hist_P))



# ====================================================================================================
#  8. DAY 2d  -  modified Newton-Raphson  (Exercise 2.4)
# ====================================================================================================
#  modified_newton_raphson  (uses internal_force, build_tangent, enforce)
#  Same loop as section 7, but Kt is built ONCE per load step (from D0n) and factorized,
#  so every equilibrium iteration is just a cheap forward-backward substitution.
# ====================================================================================================

def modified_newton_raphson(X, IX, ne, neqn, mprop, bound, P_final, nincr, imax, eps_stop, plotdof):
    # Modified Newton-Raphson (slides "Modified Newton-Raphson algorithm").
    #   for n = 1..nincr:
    #       Pn  = P^{n-1} + dP
    #       D0n = D^{n-1}
    #       Kt(D0n), enforce BC, FACTORIZE once            <-- the only difference from full NR
    #       for i = 0..imax:
    #           Ri = Rint(Di) - Pn, enforce BC on Ri
    #           stop when ||Ri|| <= eps_stop * ||P_final||
    #           dDi = -Kt(D0n)^-1 Ri                       <-- reuses the SAME factorization
    #           Di+1 = Di + dDi
    #       Dn = Di
    # The tangent is never updated inside the load step, so the corrections are not
    # quite as good as in full NR: more iterations, but each one is much cheaper.
    D  = np.zeros((neqn, 1))          # D0 = 0
    P  = np.zeros((neqn, 1))          # accumulated total load, P0 = 0
    dP = P_final / nincr
    tol = eps_stop * np.linalg.norm(P_final)

    fixed = [DOF_PER_NODE*int(node) - 2 + int(local_dof) - 1 for node, local_dof, _ in bound]

    pdof   = int(plotdof) - 1
    hist_u = [0.0]
    hist_P = [0.0]

    for n in range(1, nincr + 1):
        P = P + dP                                     # Pn = P^{n-1} + dP
                                                       # D0n = D^{n-1}  (D just carries over)
        # ---- once per load step: tangent at D0n, boundary conditions, LU factorization ----
        K = sps.csc_matrix((neqn, neqn))
        K = build_tangent(X, IX, ne, mprop, D, K)      # Kt(D0n)
        K, _ = enforce(K, np.zeros((neqn, 1)), bound)  # BC on Kt (rhs handled by R[fixed] = 0)
        lu_piv = spl.lu_factor(K.toarray())            # factorize Kt = L U  (done ONCE)

        for i in range(imax + 1):
            R = internal_force(X, IX, ne, mprop, D, neqn) - P   # Ri = Rint(Di) - Pn
            R[fixed] = 0.0                             # BC on R: no residual at supports
            res = np.linalg.norm(R)
            if res <= tol:                             # converged -> equilibrium found
                break
            if i == imax:                              # out of iterations, not converged
                print(f'  WARNING: increment {n} did not converge in {imax} iterations (||R|| = {res:.3e})')
                break

            dD = spl.lu_solve(lu_piv, -R)              # forward-backward substitution only
            D  = D + dD                                # Di+1 = Di + dDi

        print(f'  Modified NR increment {n:3d}: {i} iterations, ||R|| = {res:.3e}')

        hist_u.append(float(D[pdof].item()))           # Dn = Di
        hist_P.append(float(P[pdof].item()))

    return D, (np.array(hist_u), np.array(hist_P))



# ====================================================================================================
#  9. DAY 2 post-processing  -  analytical reference and force-displacement plot
# ====================================================================================================
#  analytical_curve, PlotForceDisplacement
# ====================================================================================================

def analytical_curve(mprop, X, IX, ne, hist):
    # Analytical single-bar reference for the UNI-AXIAL test case (Fig. 2.2):
    #   sweep strain eps, then  u = eps * L_total ,  P = A * sigma(eps).
    # L_total is the summed element length (= 3 for the two 1.5-long bars).
    # NB: this direct comparison is only meaningful for a straight uni-axial
    # bar; for a general truss the curves are not expected to match.
    A, c1, c2, c3, c4 = mprop[0]

    L_total = 0.0
    for e in range(ne):
        n1 = int(IX[e, 0]); n2 = int(IX[e, 1])
        dx = X[n2-1, 0] - X[n1-1, 0]
        dy = X[n2-1, 1] - X[n1-1, 1]
        L_total += np.sqrt(dx**2 + dy**2)

    u_fe    = hist[0]                          # FE displacements at plotdof
    eps_max = 1.05 * (max(u_fe) / L_total)     # span a bit past the FE endpoint
    eps     = np.linspace(0.0, eps_max, 400)

    u = eps * L_total
    P = A * sigma(eps, c1, c2, c3, c4)
    return u, P

def PlotForceDisplacement(curves, analytical=None):
    # Force-displacement curves at the plot DOF (the Day-2 deliverable).
    # `curves` is a list of (label, (u, P)) so several methods can be overlaid.
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



# ====================================================================================================
#  SCRIPT ENTRY POINT
# ====================================================================================================
if __name__ == '__main__':
    # Allow running this file directly (▶ Run) for debugging.
    # Change to the project root so the input file path resolves.
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    Fea('bar2.m')
