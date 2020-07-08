from photonics.sim import Tissue, Fluorescence, Detection
Tissue.populate(reserve_jobs=True)
Fluorescence.populate(reserve_jobs=True, display_progress=True)
Detection.populate(reserve_jobs=True, display_progress=True)
