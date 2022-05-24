from neurophotonics.pipeline.sim import Tissue, Fluorescence, Detection

Tissue.populate(reserve_jobs=True, display_progress=False, processes=1024)
Detection.populate(reserve_jobs=True, display_progress=False, processes=2)
Fluorescence.populate(reserve_jobs=True, display_progress=False, processes=2)

print(
    "keys processed:",
    Tissue.progress(display=False)[0]
    + Detection.progress(display=False)[0]
    + Fluorescence.progress(display=False)[0],
)
