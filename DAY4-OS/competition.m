% group18.m

X = [  0.00000  1.00000
       0.50000  0.66667
       0.50000  0.33333
       1.00000  0.00000
       1.50000  0.33333
       1.50000  0.66667
       2.00000  0.00000
       1.00000  1.00000
        ];


IX = [ 1  2  1
       1  3  1
       1  8  1
       2  3  1
       2  6  1
       2  8  1
       3  4  1
       3  5  1
       4  5  1
       4  7  1
       5  6  1
       5  7  1
       6  7  1
       6  8  1
       ];

% Element property matrix mprop = [ E A ],
mprop = [ 1.0 1.0 ];

% Prescribed loads mat(node,ldof,force),
loads = [ 1  2  -0.01];

% Boundary conditions mat(node,ldof,disp),
bound = [ 4  1  0.0
          4  2  0.0
          7  2  0.0
          ];

% Control Parameters
plotdof = 16;
