import numpy as np
import os


class MeshSetup:
    def __init__(self, level):
        self.level = level
        self.di = os.path.join(os.path.dirname(__file__), 'resources')
        try:
            assert isinstance(level, int) and 0 <= level < 11
        except AssertionError:
            raise ValueError('Invalid input. Refinement level should be an integer from 0-10.')
        self.name = 'Tohoku{:d}'.format(level)
        self.log = os.path.join(self.di, 'log')

        # Define gradations (in metres)
        inner_gradation_1 = np.linspace(7500.0, 1000.0, 11)[level]
        outer_gradation_1 = np.linspace(25000.0, 2000.0, 11)[level]
        inner_gradation_2 = np.linspace(10000.0, 2000.0, 11)[level]
        outer_gradation_2 = np.linspace(25000.0, 4000.0, 11)[level]

        # Define gradation distances (in degrees)
        gradation_distance_1 = np.linspace(0.5, 1.0, 11)[level]
        gradation_distance_2 = np.linspace(0.5, 1.0, 11)[level]

        # Parameters
        self.gradation_args_1 = (inner_gradation_1, outer_gradation_1, gradation_distance_1, 0.05)
        self.gradation_args_2 = (inner_gradation_2, outer_gradation_2, gradation_distance_2)
        self.loop_kwargs = {
            'isGlobal': False,
            'defaultPhysID': 1000,
            'fixOpenLoops': True,
        }
        self.polygon_kwargs = {
            'smallestNotMeshedArea': 5.0e+06,
            'smallestMeshedArea': 2.0e+08,
            'meshedAreaPhysID': 1,
        }
        self.mesh_kwargs = {}
        for ext in ('geo', 'fld', 'msh'):
            fname = '{:s}Filename'.format(ext)
            self.mesh_kwargs[fname] = os.path.join(self.di, 'meshes', '.'.join([self.name, ext]))

    def generate_mesh(self):
        """
        Generate mesh for Tohoku domain using QMESH.
        """
        boundary_di = os.path.join(self.di, 'boundaries')

        # Read shapefile describing domain boundaries
        boundaries = qmesh.vector.Shapes()
        boundaries.fromFile(os.path.join(boundary_di, 'boundaries.shp'))
        loop_shapes = qmesh.vector.identifyLoops(boundaries, **self.loop_kwargs)
        polygon_shapes = qmesh.vector.identifyPolygons(loop_shapes, **self.polygon_kwargs)
        polygon_shapes.writeFile(os.path.join(boundary_di, 'polygons.shp'))

        # Create raster for mesh gradation towards coastal region of importance
        fukushima_coast = qmesh.vector.Shapes()
        fukushima_coast.fromFile(os.path.join(boundary_di, 'fukushima.shp'))
        gradation_raster_fukushima = qmesh.raster.gradationToShapes()
        gradation_raster_fukushima.setShapes(fukushima_coast)
        gradation_raster_fukushima.setRasterBounds(135.0, 149.0, 30.0, 45.0)
        gradation_raster_fukushima.setRasterResolution(300, 300)
        gradation_raster_fukushima.setGradationParameters(*self.gradation_args_1)
        gradation_raster_fukushima.calculateLinearGradation()
        gradation_raster_fukushima.writeNetCDF(os.path.join(self.di, 'meshes', 'gradation_fukushima.nc'))

        # Create raster for mesh gradation towards rest of coast (Could be a polygon, line or point)
        gebco_coast = qmesh.vector.Shapes()
        gebco_coast.fromFile(os.path.join(boundary_di, 'coastline.shp'))
        gradation_raster_gebco = qmesh.raster.gradationToShapes()
        gradation_raster_gebco.setShapes(gebco_coast)
        gradation_raster_gebco.setRasterBounds(135.0, 149.0, 30.0, 45.0)
        gradation_raster_gebco.setRasterResolution(300, 300)
        gradation_raster_gebco.setGradationParameters(*self.gradation_args_2)
        gradation_raster_gebco.calculateLinearGradation()
        gradation_raster_gebco.writeNetCDF(os.path.join(self.di, 'meshes', 'gradation_gebco.nc'))

        # Create overall mesh metric
        mesh_metric_raster = qmesh.raster.meshMetricTools.minimumRaster(
            [gradation_raster_fukushima, gradation_raster_gebco])
        mesh_metric_raster.writeNetCDF(os.path.join(self.di, 'meshes', 'mesh_metric_raster.nc'))

        # Create domain object and write GMSH files
        domain = qmesh.mesh.Domain()
        domain.setTargetCoordRefSystem('EPSG:32654', fldFillValue=1000.0)
        domain.setGeometry(loop_shapes, polygon_shapes)
        domain.setMeshMetricField(mesh_metric_raster)

        # Generate GMSH file
        domain.gmsh(**self.mesh_kwargs)
        # NOTE: Default meshing algorithm is Delaunay.
        #       To use a frontal approach, include "gmshAlgo='front2d'"

    def convert_mesh(self):
        """
        Convert mesh coordinates using QMESH.
        """
        TohokuMesh = qmesh.mesh.Mesh()
        TohokuMesh.readGmsh(os.path.join(self.di, 'meshes', self.name + '.msh'), 'EPSG:3857')
        TohokuMesh.writeShapefile(os.path.join(self.di, 'meshes', self.name + '.shp'))


import qmesh
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("level", help="Integer refinement level of mesh")
args = parser.parse_args()

ms = MeshSetup(int(args.level))
qmesh.setLogOutputFile(ms.log)  # Store QMESH log for later reference
qmesh.initialise()              # Initialise QGIS API
ms.generate_mesh()              # Generate the mesh using QMESH
ms.convert_mesh()               # Convert to shapefile (for visualisation with QGIS)
