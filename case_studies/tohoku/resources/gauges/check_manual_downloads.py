import os

di = os.path.dirname(__file__)

# Temporary near-field pressure gauges, for which researchers at Tohoku University were responsible
if os.path.exists(os.path.join(di, "p02_p06.zip")):
    print("Found near-field pressure gauge data.")
else:
    raise IOError("""
        Cannot find 'p02_p06.zip'. Please contact Tatsuhiko Saito (saito-ta@bosai.go.jp) to request the
        data. Save the zip file in this directory ('{:s}') with that exact name.""".format(di))

# Pressure gauge data in the Kusiro region, for which JAMSTEC is responsible
if os.path.exists(os.path.join(di, "kusiro4698.tar.gz")):
    print("Found mid-field pressure gauge data for the Kusiro region.")
else:
    raise IOError("""
        Cannot find 'kusiro4698.tar.gz'. Please create and account at
            http://www.jamstec.go.jp/scdc/top_e.html
        and download the PG data corresponding to the Kusiro region over the appropriate time/date.
        Save the zip file in this directory ('{:s}') with that exact name.""".format(di))

# Pressure gauge data in the Muroto region, for which JAMSTEC is responsible
if os.path.exists(os.path.join(di, "muroto1655.tar.gz")):
    print("Found mid-field pressure gauge data for the Muroto region.")
else:
    raise IOError("""
        Cannot find 'muroto1655.tar.gz'. Please create and account at
            http://www.jamstec.go.jp/scdc/top_e.html
        and download the PG data corresponding to the Muroto region over the appropriate time/date.
        Save the zip file in this directory ('{:s}') with that exact name.""".format(di))

# DART far field pressure gauge data, for which NOAA is responsible
for gauge in ("21401", "21413", "21418", "21419"):
    fname = gauge + ".txt"
    if not os.path.exists(os.path.join(di, fname)):
        raise IOError("""
        Cannot find DART pressure gauge data '{:s}'. Please download it from the approprate page on
        the NOAA websity (https://www.ndbc.noaa.gov) and save it in this directory ('{:s}') with that
        exact name.""".format(fname, di))
print("Found far-field pressure gauge data.")
