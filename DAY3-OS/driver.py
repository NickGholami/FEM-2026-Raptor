# Python driver for 41525 FEM basic Matlab code
import os

# Changing to the directory of this file
os.chdir(os.path.dirname(__file__))
print ("Current working dir : %s" % os.getcwd())

# import FEA code module
from src.fea import Fea

# Define the input files to analyse (standard FEM course Matlab input format).
# Add or remove entries here to run the solver over several models in one go.
input_files = [
    "bar2.m"
]

# Perform FEA on each input file in turn
for input_file in input_files:
    print(f'\n===== Running FEA for {input_file} =====')
    fea = Fea(input_file)
