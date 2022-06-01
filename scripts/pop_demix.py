from neurophotonics.demix import *

kwargs = dict(reserve_jobs=True, display_progress=True)

IlluminationCycle.populate(**kwargs)
Demix.populate(**kwargs)
Cosine.populate(**kwargs)
SpikeSNR.populate(**kwargs)
