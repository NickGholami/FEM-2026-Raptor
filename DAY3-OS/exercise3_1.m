% Created with:       FEM Builder (FlExtract-style)
% Element type:       truss
% Number of nodes:    3
% Number of elements: 2

clear all;

% Node coordinates: x, y
X = [
0	0
1.5	-0.4
3	0
];
% Element connectivity: node1_id, node2_id, material_id
IX = [
1	2	1
2	3	1
];
// % Element properties: Young's modulus, area
mprop = [
1	1
];
% Nodal diplacements: node_id, degree of freedom (1 - x, 2 - y), displacement
bound = [
1	1	0
1	2	0
3	1	0
3	2	0
];
% Nodal loads: node_id, degree of freedom (1 - x, 2 - y), load
loads = [
2	2	0.03
];
% Control parameters
% DOF to plot: centre node 2, vertical -> 2*2 = 4
plotdof = 4;

// % Newton-Raphson parameters (Exercise 3.1: P = 0.03 in 20 increments)
nincr = 20;          % load increments
imax = 100;          % max. equilibrium iterations per increment
eps_stop = 1e-8;    % stop when ||R|| <= eps_stop * ||P_final||
spring_constant = 0.2
