from neurophotonics.design import Geometry

try:
    Geometry.populate(reserve_jobs=True, display_progress=False, processes=1024)
except ValueError:
    print("Geometry already populated")
