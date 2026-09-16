% Exercise 3.2 - Von Mises truss with a vertical spring at the centre node

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
% Element properties: Young's modulus, area
mprop = [
1	1
];
% Supports: node_id, dof (1 = x, 2 = y), displacement
bound = [
1	1	0
1	2	0
3	1	0
3	2	0
];
% Loads: node_id, dof, load
loads = [
2	2	0.03
];
% DOF to plot: centre node 2, vertical -> 2*2 = 4
plotdof = 4;

% Spring to ground at the plotted DOF
spring_constant = 0.02;

% Newton-Raphson parameters
nincr = 20;
imax = 100;
eps_stop = 1e-10;
