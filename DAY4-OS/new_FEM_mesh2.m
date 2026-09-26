% Created with:       FlExtract v1.13
% Element type:       truss
% Number of nodes:    5
% Number of elements: 7

clear all;

% Node coordinates: x, y
X = [
2.5	2.5
5	0
5	2.5
5	5
7.5	2.5
];
% Element connectivity: node1_id, node2_id, material_id
IX = [
4	1	1
3	1	1
2	1	1
5	2	1
4	3	1
5	3	1
5	4	1
];

% Prescribed loads mat(node,ldof,force)
loads = [
2	2	-0.01
];

% Element property matrix mprop = [ E A ],
mprop = [
1	1
2	2
];
% Boundary conditions mat(node,ldof,disp)   
bound = [ 1  1  0.05
          1  2  0.01
          2  1  0.0 ];


% Control Parameters
plotdof = 14;

