% Example: 2-bar straight truss
% No. of Nodes: 3
% No. of Elements: 2
clear all

% Coordinates of 3 nodes,
X = [ 0.00  0.00
      1.50  0.00
      3.00  0.00 ];

% Topology matrix IX(node1,node2,propno),
IX = [ 1  2  1
       2  3  1 ];

% Element property matrix mprop = [ c1 c2 c3 c4 A ],
mprop = [ 1.0  50.0  0.1  100.0  2.0 ];

% Unit load mat(node,ldof,force), scaled by Pfinal in Python
loads = [ 3  1  1.0 ];

% Boundary conditions mat(node,ldof,disp)
bound = [ 1  1  0.0
          1  2  0.0
          2  2  0.0
          3  2  0.0 ];

% Control Parameters
Pfinal = 200;
nincr = 20;
plotdof = 5;