h1 = 0.01;
h2 = .1;
h3 = .01;
x = 1;
r = .3;

Point(1) = {.4,-.4,0,h1};
Point(2) = {-r+.4,-.4,0,h1};
Point(3) = {.4,-r-.4,0,h1};
Point(4) = {r+.4,-.4,0,h1};
Point(5) = {.4,r-.4,0,h1};

Point(6) = {-x,-x,0,h2};
Point(7) = {x,-x,0,h2};
Point(8) = {x,x,0,h2};
Point(9) = {-x,x,0,h2};

Point(10) = {-.3, .1, 0, h3};
Point(11) = {-.3, .3, 0, h3};
Point(12) = {-.1, .3, 0, h3};
Point(13) = {-.1, .1, 0, h3};

Circle(1) = {2,1,3};
Circle(2) = {3,1,4};
Circle(3) = {4,1,5};
Circle(4) = {5,1,2};

Line(5) = {6,7};
Line(6) = {7,8};
Line(7) = {8,9};
Line(8) = {9,6};

Line(9)  = {10, 11};
Line(10) = {11, 12};
Line(11) = {12, 13};
Line(12) = {13, 10};

Line Loop(9) = {5,6,7,8};
Line Loop(10) = {1,2,3,4};
Line Loop(11) = {9,10,11,12};

Plane Surface(12) = {9,10,-11};

Physical Line(10000) = {1,2,3,4,9,10,11,12};
Physical Line(20000) = {5,6,7};
Physical Line(30000) = {8};

Physical Surface (100) = {12};
