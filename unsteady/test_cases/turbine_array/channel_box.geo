// =================================================================================================
// Gmsh geometry file for 15 turbine tidal array test case, as used in:
//   Divett, Vennell and Stevens, "Optimization of multiple turbine arrays in a channel with tidally
//   reversing flow by numerical modelling with adaptive mesh." Philosophical Transactions of the
//   Royal Society A: Mathematical, Physical and Engineering Sciences 371.1985 (2013): 20120251.
//
// Geometry file written by Nicolas Barral, 2018.
// =================================================================================================
L = 3000;
l = 1000;
//+
D = 20;
d = 5;
//+
deltax = 10*D;
deltay = 7.5*D; 
//+
dx = 100;
dxturbine = 6;
dxfarm = dxturbine;
//+
// Channel
Point(1) = {-L/2, -l/2, 0, dx};
Point(2) = {L/2, -l/2, 0, dx};
Point(3) = {L/2, l/2, 0, dx};
Point(4) = {-L/2, l/2, 0, dx};
Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 1};
Physical Line(1) = {1};
Physical Line(2) = {2};
Physical Line(3) = {3};
Physical Line(4) = {4};
Line Loop(1) = {1, 2, 3, 4};
//+ 
// turbine 1
Point(5) = {-d/2-2*deltax, -D/2+deltay, 0, dxturbine};
Point(6) = {d/2-2*deltax, -D/2+deltay, 0, dxturbine};
Point(7) = {d/2-2*deltax, D/2+deltay, 0, dxturbine};
Point(8) = {-d/2-2*deltax, D/2+deltay, 0, dxturbine};
Line(5) = {5, 6};
Line(6) = {6, 7};
Line(7) = {7, 8};
Line(8) = {8, 5};
//Physical Line(5) = {5};
//Physical Line(6) = {6};
//Physical Line(7) = {7};
//Physical Line(8) = {8};
Line Loop(2) = {5, 6, 7, 8};
// turbine 2
Point(9 ) = {-d/2-2*deltax, -D/2, 0, dxturbine};
Point(10) = {d/2-2*deltax, -D/2, 0, dxturbine};
Point(11) = {d/2-2*deltax, D/2, 0, dxturbine};
Point(12) = {-d/2-2*deltax, D/2, 0, dxturbine};
Line(9 ) = {9 , 10};
Line(10) = {10, 11};
Line(11) = {11, 12};
Line(12) = {12, 9};
//Physical Line(9 ) = {9 };
//Physical Line(10) = {10};
//Physical Line(11) = {11};
//Physical Line(12) = {12};
Line Loop(3) = {9, 10, 11, 12};
// turbine 3
Point(13) = {-d/2-2*deltax, -D/2-deltay, 0, dxturbine};
Point(14) = {d/2-2*deltax, -D/2-deltay, 0, dxturbine};
Point(15) = {d/2-2*deltax, D/2-deltay, 0, dxturbine};
Point(16) = {-d/2-2*deltax, D/2-deltay, 0, dxturbine};
Line(13) = {13, 14};
Line(14) = {14, 15};
Line(15) = {15, 16};
Line(16) = {16, 13};
//Physical Line(13) = {13};
//Physical Line(14) = {14};
//Physical Line(15) = {15};
//Physical Line(16) = {16};
Line Loop(4) = {13, 14, 15, 16};
// turbine 4
Point(17) = {-d/2-deltax, -D/2+deltay, 0, dxturbine};
Point(18) = {d/2-deltax, -D/2+deltay, 0, dxturbine};
Point(19) = {d/2-deltax, D/2+deltay, 0, dxturbine};
Point(20) = {-d/2-deltax, D/2+deltay, 0, dxturbine};
Line(17) = {17, 18};
Line(18) = {18, 19};
Line(19) = {19, 20};
Line(20) = {20, 17};
//Physical Line(17) = {17};
//Physical Line(18) = {18};
//Physical Line(19) = {19};
//Physical Line(20) = {20};
Line Loop(5) = {17, 18, 19, 20};
// turbine 5
Point(21) = {-d/2-deltax, -D/2, 0, dxturbine};
Point(22) = {d/2-deltax, -D/2, 0, dxturbine};
Point(23) = {d/2-deltax, D/2, 0, dxturbine};
Point(24) = {-d/2-deltax, D/2, 0, dxturbine};
Line(21) = {21, 22};
Line(22) = {22, 23};
Line(23) = {23, 24};
Line(24) = {24, 21};
//Physical Line(21) = {21};
//Physical Line(22) = {22};
//Physical Line(23) = {23};
//Physical Line(24) = {24};
Line Loop(6) = {21, 22, 23, 24};
// turbine 6
Point(25) = {-d/2-deltax, -D/2-deltay, 0, dxturbine};
Point(26) = {d/2-deltax, -D/2-deltay, 0, dxturbine};
Point(27) = {d/2-deltax, D/2-deltay, 0, dxturbine};
Point(28) = {-d/2-deltax, D/2-deltay, 0, dxturbine};
Line(25) = {25, 26};
Line(26) = {26, 27};
Line(27) = {27, 28};
Line(28) = {28, 25};
//Physical Line(25) = {25};
//Physical Line(26) = {26};
//Physical Line(27) = {27};
//Physical Line(28) = {28};
Line Loop(7) = {25, 26, 27, 28};
// turbine 7
Point(29) = {-d/2, -D/2+deltay, 0, dxturbine};
Point(30) = {d/2, -D/2+deltay, 0, dxturbine};
Point(31) = {d/2, D/2+deltay, 0, dxturbine};
Point(32) = {-d/2, D/2+deltay, 0, dxturbine};
Line(29) = {29, 30};
Line(30) = {30, 31};
Line(31) = {31, 32};
Line(32) = {32, 29};
//Physical Line(29) = {29};
//Physical Line(30) = {30};
//Physical Line(31) = {31};
//Physical Line(32) = {32};
Line Loop(8) = {29, 30, 31, 32};
// turbine 8
Point(33) = {-d/2, -D/2, 0, dxturbine};
Point(34) = {d/2, -D/2, 0, dxturbine};
Point(35) = {d/2, D/2, 0, dxturbine};
Point(36) = {-d/2, D/2, 0, dxturbine};
Line(33) = {33, 34};
Line(34) = {34, 35};
Line(35) = {35, 36};
Line(36) = {36, 33};
//Physical Line(33) = {33};
//Physical Line(34) = {34};
//Physical Line(35) = {35};
//Physical Line(36) = {36};
Line Loop(9) = {33, 34, 35, 36};
// turbine 9
Point(37) = {-d/2, -D/2-deltay, 0, dxturbine};
Point(38) = {d/2, -D/2-deltay, 0, dxturbine};
Point(39) = {d/2, D/2-deltay, 0, dxturbine};
Point(40) = {-d/2, D/2-deltay, 0, dxturbine};
Line(37) = {37, 38};
Line(38) = {38, 39};
Line(39) = {39, 40};
Line(40) = {40, 37};
//Physical Line(37) = {37};
//Physical Line(38) = {38};
//Physical Line(39) = {39};
//Physical Line(40) = {40};
Line Loop(10) = {37, 38, 39, 40};
// turbine 10
Point(41) = {-d/2+deltax, -D/2+deltay, 0, dxturbine};
Point(42) = {d/2+deltax, -D/2+deltay, 0, dxturbine};
Point(43) = {d/2+deltax, D/2+deltay, 0, dxturbine};
Point(44) = {-d/2+deltax, D/2+deltay, 0, dxturbine};
Line(41) = {41, 42};
Line(42) = {42, 43};
Line(43) = {43, 44};
Line(44) = {44, 41};
//Physical Line(41) = {41};
//Physical Line(42) = {42};
//Physical Line(43) = {43};
//Physical Line(44) = {44};
Line Loop(11) = {41, 42, 43, 44};
// turbine 11
Point(45) = {-d/2+deltax, -D/2, 0, dxturbine};
Point(46) = {d/2+deltax, -D/2, 0, dxturbine};
Point(47) = {d/2+deltax, D/2, 0, dxturbine};
Point(48) = {-d/2+deltax, D/2, 0, dxturbine};
Line(45) = {45, 46};
Line(46) = {46, 47};
Line(47) = {47, 48};
Line(48) = {48, 45};
//Physical Line(45) = {45};
//Physical Line(46) = {46};
//Physical Line(47) = {47};
//Physical Line(48) = {48};
Line Loop(12) = {45, 46, 47, 48};
// turbine 12
Point(49) = {-d/2+deltax, -D/2-deltay, 0, dxturbine};
Point(50) = {d/2+deltax, -D/2-deltay, 0, dxturbine};
Point(51) = {d/2+deltax, D/2-deltay, 0, dxturbine};
Point(52) = {-d/2+deltax, D/2-deltay, 0, dxturbine};
Line(49) = {49, 50};
Line(50) = {50, 51};
Line(51) = {51, 52};
Line(52) = {52, 49};
//Physical Line(49) = {49};
//Physical Line(50) = {50};
//Physical Line(51) = {51};
//Physical Line(52) = {52};
Line Loop(13) = {49, 50, 51, 52};
// turbine 13
Point(53) = {-d/2+2*deltax, -D/2+deltay, 0, dxturbine};
Point(54) = {d/2+2*deltax, -D/2+deltay, 0, dxturbine};
Point(55) = {d/2+2*deltax, D/2+deltay, 0, dxturbine};
Point(56) = {-d/2+2*deltax, D/2+deltay, 0, dxturbine};
Line(53) = {53, 54};
Line(54) = {54, 55};
Line(55) = {55, 56};
Line(56) = {56, 53};
//Physical Line(53) = {53};
//Physical Line(54) = {54};
//Physical Line(55) = {55};
//Physical Line(56) = {56};
Line Loop(14) = {53, 54, 55, 56};
// turbine 14
Point(57) = {-d/2+2*deltax, -D/2, 0, dxturbine};
Point(58) = {d/2+2*deltax, -D/2, 0, dxturbine};
Point(59) = {d/2+2*deltax, D/2, 0, dxturbine};
Point(60) = {-d/2+2*deltax, D/2, 0, dxturbine};
Line(57) = {57, 58};
Line(58) = {58, 59};
Line(59) = {59, 60};
Line(60) = {60, 57};
//Physical Line(57) = {57};
//Physical Line(58) = {58};
//Physical Line(59) = {59};
//Physical Line(60) = {60};
Line Loop(15) = {57, 58, 59, 60};
// turbine 15
Point(61) = {-d/2+2*deltax, -D/2-deltay, 0, dxturbine};
Point(62) = {d/2+2*deltax, -D/2-deltay, 0, dxturbine};
Point(63) = {d/2+2*deltax, D/2-deltay, 0, dxturbine};
Point(64) = {-d/2+2*deltax, D/2-deltay, 0, dxturbine};
Line(61) = {61, 62};
Line(62) = {62, 63};
Line(63) = {63, 64};
Line(64) = {64, 61};
//Physical Line(61) = {61};
//Physical Line(62) = {62};
//Physical Line(63) = {63};
//Physical Line(64) = {64};
Line Loop(16) = {61, 62, 63, 64};
//+
// Refined area around the turbines
Point(65) = {-3*deltax, -1.3*deltay, 0, dxfarm};
Point(66) = {3*deltax, -1.3*deltay, 0, dxfarm};
Point(67) = {3*deltax, 1.3*deltay, 0, dxfarm};
Point(68) = {-3*deltax, 1.3*deltay, 0, dxfarm};
Line(65) = {65, 66};
Line(66) = {66, 67};
Line(67) = {67, 68};
Line(68) = {68, 65};
Line Loop(17) = {65, 66, 67, 68};
//+
Plane Surface(1) = {1, 17};
Plane Surface(2 ) = {2 };
Plane Surface(3 ) = {3 };
Plane Surface(4 ) = {4 };
Plane Surface(5 ) = {5 };
Plane Surface(6 ) = {6 };
Plane Surface(7 ) = {7 };
Plane Surface(8 ) = {8 };
Plane Surface(9 ) = {9 };
Plane Surface(10) = {10};
Plane Surface(11) = {11};
Plane Surface(12) = {12};
Plane Surface(13) = {13};
Plane Surface(14) = {14};
Plane Surface(15) = {15};
Plane Surface(16) = {16};
Plane Surface(17) = {17, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
//+
Physical Surface(1 ) = {1, 17 };
Physical Surface(2 ) = {2 };
Physical Surface(3 ) = {3 };
Physical Surface(4 ) = {4 };
Physical Surface(5 ) = {5 };
Physical Surface(6 ) = {6 };
Physical Surface(7 ) = {7 };
Physical Surface(8 ) = {8 };
Physical Surface(9 ) = {9 };
Physical Surface(10) = {10};
Physical Surface(11) = {11};
Physical Surface(12) = {12};
Physical Surface(13) = {13};
Physical Surface(14) = {14};
Physical Surface(15) = {15};
Physical Surface(16) = {16};
