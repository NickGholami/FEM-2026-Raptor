% group18.m
% Optimized truss for the Day 1 stiffness competition.
% A = E = 1, total element length <= 10.

X = [
    0.00000000    1.00000000
    0.48386503    0.97470844
    0.50000000    0.33333333
    1.00000000    0.00000000
    1.50000000    0.33333333
    1.71818502    0.70554582
    2.00000000    0.00000000
    0.50000000    1.00000000
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

% Element properties: Young's modulus E, area A
mprop = [
    1.00000000    1.00000000
];

% Prescribed load: node, local DOF, force
loads = [
    1    2   -0.01000000
];

% Boundary conditions: node, local DOF, displacement
bound = [
    4    1    0.00000000
    4    2    0.00000000
    7    2    0.00000000
];

plotdof = 16;
