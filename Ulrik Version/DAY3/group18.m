% group18.m

X = [
    0.00000000    1.00000000
    0.49999733    0.82331912
    0.49999973    0.33333306
    1.00000000    0.00000000
    1.50000056    0.33333292
    1.67122182    0.66914130
    2.00000000    0.00000000
    0.76000221    0.99552679
];

IX = [
    1    2    1
    1    3    1
    1    8    1
    2    3    1
    2    6    1
    2    8    1
    3    4    1
    3    5    1
    4    5    1
    4    7    1
    5    6    1
    5    7    1
    6    7    1
    6    8    1
];

% Element property matrix mprop = [ E A ],
mprop = [
    1.00000000    1.00000000
];

% Prescribed loads mat(node,ldof,force),
loads = [
    1    2    -0.01000000
];

% Boundary conditions mat(node,ldof,disp),
bound = [
    4    1    0.00000000
    4    2    0.00000000
    7    2    0.00000000
];

% Control Parameters
plotdof = 16;
