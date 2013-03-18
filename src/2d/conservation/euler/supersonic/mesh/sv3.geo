lc =0.025;
Point(1) = {0.0,1.0,0,lc};
Point(2) = {0.0,1.384,0,lc};
Point(3) = {1.384,0.0,0,lc};
Point(4) = {1.0,0.0,0.0,lc};
Point(5) = {0.0,0.0,0.0,lc};

Line(1) = {2,1};
Line(2) = {4,3};
Circle(3) = {4,5,1};
Circle(4) = {3,5,2};

Line Loop(5) = {1,-3,2,4};
Plane Surface(6) = {5};

Physical Surface (100) = {6};
Physical Line (20000) = {3,4};






