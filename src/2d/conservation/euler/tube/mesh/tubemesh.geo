//horn points
//Point(1) = {-0.26,0,0,charLen5};

h = 0.0025;
charLen5 = 0.0019;
charLen6 = 0.002;
charLen7 = 0.0075;
charLen8 = 0.005;

Point(2) = {0,0.0075,0,h};
Point(3) = {0,-0.0075,0,h};


Point(16) = {4.,0.0075,0,h};
Point(17) = {4.,-0.0075,0,h};

Line(1) = {2,3};
Line(2) = {3,17};
Line(3) = {17,16};
Line(4) = {16,2};

Line Loop(100) = {1,2,3,4};
Plane Surface(102) ={100};
Physical Surface(103) = {102};
Physical Line(10000) = {2,4}; 
Physical Line(30000) = {1,3};
