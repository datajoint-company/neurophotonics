from neurophotonics.sim import Tissue, Fluorescence, Detection
Tissue.populate(reserve_jobs=True)
Detection.populate(reserve_jobs=True, display_progress=True)
Fluorescence.populate(reserve_jobs=True, display_progress=True)
