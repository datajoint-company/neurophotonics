from neurophotonics.demix import *

IlluminationCycle.populate(reserve_jobs=True, processes=1024, display_progress=False)
Demix.populate('sample<=5', display_progress=False, processes=1024, reserve_jobs=True)
Cosine.populate(reserve_jobs=True, display_progress=False, processes=1024)
SpikeSNR.populate(reserve_jobs=True, display_progress=False, processes=1024)
