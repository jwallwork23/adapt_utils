// Basin dimension
basin_x = 1.5*13800;
basin_y = 1200;
// Number of elements
nx = basin_x/1200; 
ny = basin_y/1200;

Point(1) = {0, 0, 0};
Extrude{basin_x, 0, 0} { Point{1}; Layers{nx}; }
Extrude{0, basin_y, 0} { Line{1}; Layers{ny}; }

Physical Line(1) = {3};
Physical Line(3) = {1};
Physical Line(2) = {4};
Physical Line(4) = {2};
Physical Surface(6) = {5};
