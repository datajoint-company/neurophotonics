from neurophotonics.sim import Tissue, Fluorescence, Detection

Tissue.populate(reserve_jobs=True, display_progress=False, processes=1024)
Detection.populate(reserve_jobs=True, display_progress=False, processes=1024)
Fluorescence.populate(reserve_jobs=True, display_progress=False, processes=1024)
