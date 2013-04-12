charLen1= 0.002;
charLen2 = 0.09;
charLen3 = 0.09;
charLen4 = 0.2;
charLen5 = 0.0009;
charLen10 = 0.08;

a=0.09; //Jan30mesh1.msh (order magnitude too small) has a=0.5, Jan30mesh2.msh has a=0.09
c=0.01;

Point(1) = {148*c,0,0,charLen1};
Point(2) = {148*c,6.5*c,0,charLen5};
Point(3) = {148*c,-6.5*c,0,charLen5};

Point(4) = {147.5*c,0,0,charLen1};
Point(5) = {147.5*c,5.875*c,0,charLen1};
Point(6) = {147.5*c,-5.875*c,0,charLen1};

Point(7) = {145*c,0,0,charLen1};
Point(8) = {145*c,2.93*c,0,charLen1};
Point(9) = {145*c,-2.93*c,0,charLen1};

Point(10) = {142*c,0,0,charLen1};
Point(11) = {142*c,2.125*c,0,charLen1};
Point(12) = {142*c,-2.125*c,0,charLen1};

Point(13) = {138*c,0,0,charLen1};
Point(14) = {138*c,1.475*c,0,charLen1};
Point(15) = {138*c,-1.475*c,0,charLen1};

Point(16) = {135*c,0,0,charLen1};
Point(17) = {135*c,1.325*c,0,charLen1};
Point(18) = {135*c,-1.325*c,0,charLen1};

Point(19) = {132*c,0,0,charLen1};
Point(20) = {132*c,1.225*c,0,charLen1};
Point(21) = {132*c,-1.225*c,0,charLen1};

Point(22) = {128.5*c,0,0,charLen1};
Point(23) = {128.5*c,1.125*c,0,charLen1};
Point(24) = {128.5*c,-1.125*c,0,charLen1};

Point(25) = {122*c,0,0,charLen1};
Point(26) = {122*c,0.975*c,0,charLen1};
Point(27) = {122*c,-0.975*c,0,charLen1};

Point(28) = {116*c,0,0,charLen1};
Point(29) = {116*c,0.85*c,0,charLen1};
Point(30) = {116*c,-0.85*c,0,charLen1};

Point(31) = {109.5*c,0,0,charLen1};
Point(32) = {109.5*c,0.75*c,0,charLen1};
Point(33) = {109.5*c,-0.75*c,0,charLen1};

Point(34) = {102*c,0,0,charLen1};
Point(35) = {102*c,0.7*c,0,charLen1};
Point(36) = {102*c,-0.7*c,0,charLen1};

//tube points
Point(37) = {0,0,0,charLen1};
Point(38) = {0,0.7*c,0,charLen1};
Point(39) = {0,-0.7*c,0,charLen1};


Line(1000) ={2,3};
Spline(1) = {2,5,8,11,14,17,20,23,26,29,32,35};
//Spline(2) = {35,37};
//Spline(3) = {37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75};
//Spline(4) = {36,75};
Spline(5) = {3,6,9,12,15,18,21,24,27,30,33,36};

Line(2) = {35,38};
Line(5000) = {38,39};
Line(4) = {36,39};

////////////////////////////////////


Point(102) = {148*c,(6.5+1*a)*c,0,charLen5};
Point(103) = {148*c,(-6.5-1*a)*c,0,charLen5};

Point(105) = {147.5*c,(5.875+a)*c,0,charLen5};
Point(106) = {147.5*c,(-5.875-a)*c,0,charLen5};

Point(108) = {145*c,(2.93+a)*c,0,charLen5};
Point(109) = {145*c,(-2.93-a)*c,0,charLen5};

Point(111) = {142*c,(2.125+a)*c,0,charLen5};
Point(112) = {142*c,(-2.125-a)*c,0,charLen5};

Point(114) = {138*c,(1.475+a)*c,0,charLen1};
Point(115) = {138*c,(-1.475-a)*c,0,charLen1};

Point(117) = {135*c,(1.325+a)*c,0,charLen1};
Point(118) = {135*c,(-1.325-a)*c,0,charLen1};

Point(120) = {132*c,(1.225+a)*c,0,charLen3};
Point(121) = {132*c,(-1.225-a)*c,0,charLen3};

Point(123) = {128.5*c,(1.125+a)*c,0,charLen3};
Point(124) = {128.5*c,(-1.125-a)*c,0,charLen3};

Point(126) = {122*c,(0.975+a)*c,0,charLen2};
Point(127) = {122*c,(-0.975-a)*c,0,charLen2};

Point(129) = {116*c,(0.85+a)*c,0,charLen2};
Point(130) = {116*c,(-0.85-a)*c,0,charLen2};

Point(132) = {109.5*c,(0.75+a)*c,0,charLen2};
Point(133) = {109.5*c,(-0.75-a)*c,0,charLen2};

Point(135) = {102*c,(0.7+a)*c,0,charLen2};
Point(136) = {102*c,(-0.7-a)*c,0,charLen2};

//tube points

Point(138) = {(0-a)*c,(0.7+a)*c,0,charLen2};
Point(139) = {(0-a)*c,(-0.7-a)*c,0,charLen2};

Line(2000) ={102,2};
Line(3000)={3,103};

Spline(101) = {102,105,108,111,114,117,120,123,126,129,132,135};
//Spline(102) = {135,137};
//Spline(103) = {137,138,139,140,141,142,143,144,145,146,147,148,149,150,151,152,153,154,155,156,157,158,159,160,161,162,163,164,165,166,167,168,169,170,171,172,173,174,175};
//Spline(104) = {136,175};
Spline(105) = {103,106,109,112,115,118,121,124,127,130,133,136};


Line(102) = {135,138};
Line(6000) = {138,139};
Line(104) = {136,139};

/////////////////



Point(80) = {320*c,-150*c,0,charLen4};
Point(81) = {-4*c,-4*c,0,charLen2};
Point(82) = {320*c,150*c,0,charLen4};
Point(83) = {-4*c,4*c,0,charLen2};
Point(84) = {100*c,4*c,0,charLen10};
Point(85) = {100*c,-4*c,0,charLen10};
Point(86) = {100*c,150*c,0,charLen4};
Point(87) = {100*c,-150*c,0,charLen4};

Line(6) = {82,86};
Line(7) = {86,84};
Line(8) = {84,83};
Line(9) = {83,81};
Line(10) = {85,81};
Line(11) = {85,87};
Line(12) = {80,87};
Line(13) = {82,80};


//////////////////////////////////////////////
//Make physical lines


//Physical Line(40000)={6000,2000,3000};
//Physical Line(200) ={6,7,8,9,10,11,12,13};
//Physical Line(20000) = {1,2,4,5,101,102,104,105};

// 20000 reflecting, 40000 inflow, 200 ???
// old labeling convention
//Physical Line(40000)={6000,2000,3000,2,4,102,104};
//Physical Line(200) ={6,7,8,9,10,11,12,13};
//Physical Line(20000) = {1,5,101,105};

// new labeling convention
Physical Line(30000) = {5000, 9, 8, 7, 6, 13, 12, 11, 10};
Physical Line(10000) = {2, 102, 1, 101, 5, 105, 4, 104, 6000};


Line Loop(17) = {1,2,5000,-4,-5,-1000};//,1,2,3,-4,-5}; //border of entire trumpet  //added 6, 106 not there (like in old meshes)
Line Loop(16) = {1,2,5000,-4,-5,3000,105,104,-6000,-102,-101,2000};

Plane Surface(102)= {17};

Line Loop(22) = {6,7,8,9,-10,11,-12,-13}; //box
Line Loop(23) = {-2000,101,102,6000,-104,-105,-3000,-1000}; //outside boundary of trumpet actual

Plane Surface(100)={22,23};
Physical Surface(103) = {100,102};

