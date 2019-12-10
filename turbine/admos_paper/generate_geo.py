from adapt_utils.turbine.meshgen import *
from adapt_utils.turbine.options import *

generate_geo_file(Steady2TurbineOptions(), level='xcoarse')
generate_geo_file(Steady2TurbineOptions(), level='coarse')
generate_geo_file(Steady2TurbineOptions(), level='medium')
generate_geo_file(Steady2TurbineOptions(), level='fine')
generate_geo_file(Steady2TurbineOptions(), level='xfine')
generate_geo_file(Steady2TurbineOffsetOptions(), level='xcoarse', tag='offset')
generate_geo_file(Steady2TurbineOffsetOptions(), level='coarse', tag='offset')
generate_geo_file(Steady2TurbineOffsetOptions(), level='medium', tag='offset')
generate_geo_file(Steady2TurbineOffsetOptions(), level='fine', tag='offset')
generate_geo_file(Steady2TurbineOffsetOptions(), level='xfine', tag='offset')
