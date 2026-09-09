% bar2.m  -  Exercise 2.1, the simple 2-bar test specimen (Figure 2.2)
%
% Three nodes on a straight horizontal line, total length 3 (1.5 + 1.5).
% Node 1 is pinned (x and y).  Nodes 2 and 3 are rollers (y fixed) so the
% whole thing can only move in x -> a pure uni-axial test, exactly the case
% the analytical curve in analytical_bar.py describes.

% Node coordinates  (x, y)
X = [
    0.0    0.0
    1.5    0.0
    3.0    0.0
];

% Topology  IX(node1, node2, propno)
IX = [
    1    2    1
    2    3    1
];

// % Material properties for the rubber (Signorini) material:
// %   mprop = [ A  c1  c2  c3  c4 ]
mprop = [
    2.0    1.0    50.0    0.1    100.0
];

% Prescribed load: node, local DOF (1=x, 2=y), FINAL force value
loads = [
    3    1    200.0
];

% Boundary conditions: node, local DOF, prescribed displacement
bound = [
    1    1    0.0
    1    2    0.0
    2    2    0.0
    3    2    0.0
];

// % DOF to plot the force-displacement curve for (node 3, x):  2*3-1 = 5
plotdof = 5;

% Euler incremental method parameters (given in the input file, as required)
Pfinal = 200;
nincr = 20;

% Solver choice:  1 = pure Euler,  2 = Euler + equilibrium-correction.
% Leave this line out to run BOTH and overlay them for comparison.
method = 2;
