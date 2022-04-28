from neurophotonics.demix import *

try:
    IlluminationCycle.populate(reserve_jobs=True, processes=1024, display_progress=False)
except ValueError:
    print("IlluminationCycle already populated")

try:
    Demix.populate(display_progress=False, processes=1024, reserve_jobs=True)
except ValueError:
    print("Demix already populated")

try:
    Cosine.populate(reserve_jobs=True, display_progress=False, processes=1024)
except ValueError:
    print("Cosine already populated")


try:
    SpikeSNR.populate(reserve_jobs=True, display_progress=False, processes=1024)
except ValueError:
    print("SpikeSNR already populated")
