"""
Generate a hierarchy of meshes for the Tohoku tsunami problem.
"""
import numpy as np
import os
import qmesh


class MeshSetup:
    """
    Generate mesh `level` in the hierarchy.

    Whilst we do not have a nested hierarchy, the gradation parameters are chosen such that
    the mesh on level n+1 has approximately four times as many elements as the mesh on level n.
    """
    def __init__(self, level):
        self.level = level
        self.di = os.path.join(os.path.dirname(__file__), 'resources')
        if not (isinstance(level, int) and level >= 0):
            raise ValueError('Invalid input. Refinement level should be a non-negative integer')
        self.name = 'Tohoku{:d}'.format(level)
        self.log = os.path.join(self.di, 'meshes', 'log')

        # Gradation parameters for section of coast near Fukushima
        self.gradation_args_fukushima = (
            7.5e+03*0.5**level,   # inner gradation (m)
            30.0e+03*0.5**level,  # outer gradation (m)
            1.0,                  # gradation distance (degrees)
            0.05,                 # TODO: What is this parameter?
        )

        # Gradation parameters for the rest of the coast
        self.gradation_args_else = (
            10.0e+03*0.5**level,  # inner gradation (m)
            30.0e+03*0.5**level,  # outer gradation (m)
            0.5,                  # gradation distance (degrees)
        )

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
        gradation_raster_fukushima.setGradationParameters(*self.gradation_args_fukushima)
        gradation_raster_fukushima.calculateLinearGradation()
        gradation_raster_fukushima.writeNetCDF(os.path.join(self.di, 'meshes', 'gradation_fukushima.nc'))

        # Create raster for mesh gradation towards rest of coast (Could be a polygon, line or point)
        gebco_coast = qmesh.vector.Shapes()
        gebco_coast.fromFile(os.path.join(boundary_di, 'coastline.shp'))
        gradation_raster_gebco = qmesh.raster.gradationToShapes()
        gradation_raster_gebco.setShapes(gebco_coast)
        gradation_raster_gebco.setRasterBounds(135.0, 149.0, 30.0, 45.0)
        gradation_raster_gebco.setRasterResolution(300, 300)
        gradation_raster_gebco.setGradationParameters(*self.gradation_args_else)
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


for i in range(5):
    ms = MeshSetup(i)
    qmesh.setLogOutputFile(ms.log)  # Store QMESH log for later reference
    qmesh.initialise()              # Initialise QGIS API
    ms.generate_mesh()              # Generate the mesh using QMESH
    ms.convert_mesh()               # Convert to shapefile (for visualisation with QGIS)
