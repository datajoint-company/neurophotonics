from photonics.demix import *

IlluminationCycle.populate(reserve_jobs=True, display_progress=True)
Demix.populate('sample<=5', display_progress=True, reserve_jobs=True)
