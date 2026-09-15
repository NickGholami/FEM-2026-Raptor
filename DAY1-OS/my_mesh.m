% Created with:       FEM Builder (FlExtract-style)
% Element type:       truss
% Number of nodes:    16
% Number of elements: 32

clear all;

% Node coordinates: x, y
X = [
0	0.25
0	0.5
0	0.75
0	1
0.25	0
0.25	0.25
0.25	0.5
0.25	0.75
0.25	1
0	0
0.5	0.75
0.75	0.75
1	0.75
1	1
0.75	1
0.5	1
];
% Element connectivity: node1_id, node2_id, material_id
IX = [
10	5	1
5	6	1
6	7	1
7	8	1
8	9	1
9	16	1
15	16	1
14	15	1
8	11	1
11	12	1
12	13	1
13	14	1
4	9	1
4	3	1
8	3	1
2	3	1
2	1	1
1	10	1
6	1	1
2	7	1
5	1	1
1	7	1
7	3	1
3	9	1
8	4	1
9	11	1
11	16	1
16	12	1
12	15	1
15	11	1
15	13	1
12	14	1
];
% Element properties: Young's modulus, area
mprop = [
70000000000	0.0002
];
% Nodal diplacements: node_id, degree of freedom (1 - x, 2 - y), displacement
bound = [
10	2	0
10	1	0
5	2	0
];
% Nodal loads: node_id, degree of freedom (1 - x, 2 - y), load
loads = [
13	2	-10000
];
% Control parameters
plotdof = 32;
