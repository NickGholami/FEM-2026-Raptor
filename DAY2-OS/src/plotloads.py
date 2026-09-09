# This module plots loads from the FE problem

import numpy             as np
import matplotlib.pyplot as plt

def plotloads(loads, Xnew, dsup):
        # using dsup from supports to dimension the arrows to the structure
        dsup0 = abs(dsup)*3
        # finding the max load
        maxload = np.max(np.abs(loads[:, 2]))

        # Colour of the force arrows (kept distinct from red/green/blue used for
        # compression/non-loaded/tension). Change this to restyle all load arrows.
        fcol = 'm'  # magenta

        nload = loads.shape[0]
        for L in range(nload):
            # To give them the right direction and relative size
            dsup = abs(dsup0) * np.sign(loads[L, 2]) * abs(loads[L, 2]) / maxload
            XX = Xnew[int(loads[L, 0])-1, 0]
            YY = Xnew[int(loads[L, 0])-1, 1]
            if loads[L, 1] == 1:  # horizontal force
                plt.plot([XX, XX + 3*dsup], [YY, YY], fcol, linewidth=2)
                plt.plot([XX + 3*dsup, XX + 2*dsup], [YY, YY + 0.7*dsup], fcol, linewidth=2)
                plt.plot([XX + 3*dsup, XX + 2*dsup], [YY, YY - 0.7*dsup], fcol, linewidth=2)
            elif loads[L, 1] == 2:  # vertical force
                plt.plot([XX, XX], [YY, YY + 3*dsup], fcol, linewidth=2)
                plt.plot([XX, XX + 0.7*dsup], [YY + 3*dsup, YY + 2*dsup], fcol, linewidth=2)
                plt.plot([XX, XX - 0.7*dsup], [YY + 3*dsup, YY + 2*dsup], fcol, linewidth=2)