h1 = .01;
Point(1) = {.5,1,0,h1};
Point(2) = {0,0,0,h1};
Point(3) = {0,2,0,h1};
Point(4) = {.5,2,0,h1};
Point(5) = {.5,0,0,h1};
Point(6) = {1.5,1,0,h1};

Circle(1) = {4,1,6};
Circle(2) = {6,1,5};

Line(3) = {2,3};
Line(4) = {3,4};
Line(5) = {5,2};

Line Loop(5) = {1,2,3,4,5};
Plane Surface(6) = {-5};

Physical Surface (100) = {6};
Physical Line(10000) = {1,2,4,5};
Physical Line(30000) = {3};
