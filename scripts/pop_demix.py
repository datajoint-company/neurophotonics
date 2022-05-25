from neurophotonics.demix import *

IlluminationCycle.populate(reserve_jobs=True, processes=2, display_progress=False)
Demix.populate(display_progress=False, processes=1024, reserve_jobs=True)
Cosine.populate(reserve_jobs=True, display_progress=False, processes=1)
SpikeSNR.populate(reserve_jobs=True, display_progress=False, processes=1)

print(
    "keys processed:",
    IlluminationCycle.progress(display=False)[0]
    + Demix.progress(display=False)[0]
    + Cosine.progress(display=False)[0]
    + SpikeSNR.progress(display=False)[0],
)
