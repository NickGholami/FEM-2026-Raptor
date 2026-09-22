% Created with:       FlExtract v1.13
% Element type:       truss
% Number of nodes:    42
% Number of elements: 101

clear all;

% Node coordinates: x, y
X = [
0	0
0	1
1	0
1	1
2	0
2	1
3	0
3	1
4	0
4	1
5	0
5	1
6	0
6	1
7	0
7	1
8	0
8	1
9	0
9	1
10	0
10	1
11	0
11	1
12	0
12	1
13	0
13	1
14	0
14	1
15	0
15	1
16	0
16	1
17	0
17	1
18	0
18	1
19	0
19	1
20	0
20	1
];
% Element connectivity: node1_id, node2_id, material_id
IX = [
2	1	1
4	1	1
3	1	1
4	2	1
3	2	1
4	3	1
6	3	1
5	3	1
6	4	1
5	4	1
6	5	1
8	5	1
7	5	1
8	6	1
7	6	1
8	7	1
10	7	1
9	7	1
10	8	1
9	8	1
10	9	1
12	9	1
11	9	1
12	10	1
11	10	1
12	11	1
14	11	1
13	11	1
14	12	1
13	12	1
14	13	1
16	13	1
15	13	1
16	14	1
15	14	1
16	15	1
18	15	1
17	15	1
18	16	1
17	16	1
18	17	1
20	17	1
19	17	1
20	18	1
19	18	1
20	19	1
22	19	1
21	19	1
22	20	1
21	20	1
22	21	1
24	21	1
23	21	1
24	22	1
23	22	1
24	23	1
26	23	1
25	23	1
26	24	1
25	24	1
26	25	1
28	25	1
27	25	1
28	26	1
27	26	1
28	27	1
30	27	1
29	27	1
30	28	1
29	28	1
30	29	1
32	29	1
31	29	1
32	30	1
31	30	1
32	31	1
34	31	1
33	31	1
34	32	1
33	32	1
34	33	1
36	33	1
35	33	1
36	34	1
35	34	1
36	35	1
38	35	1
37	35	1
38	36	1
37	36	1
38	37	1
40	37	1
39	37	1
40	38	1
39	38	1
40	39	1
42	39	1
41	39	1
42	40	1
41	40	1
42	41	1
];
// % Element properties: Young's modulus, area
mprop = [
1	1
2	2
];
% Nodal diplacements: node_id, degree of freedom (1 - x, 2 - y), displacement
bound = [
1	1	0
1	2	0
2	1	0
2	2	0
];
% Nodal loads: node_id, degree of freedom (1 - x, 2 - y), load
loads = [
41	2	-1e-6
42	2	-1e-6
];
% Control parameters
plotdof = 82;
nincr = 20;
imax = 100;
