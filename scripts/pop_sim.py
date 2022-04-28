from neurophotonics.sim import Tissue, Fluorescence, Detection

try:
    Tissue.populate(reserve_jobs=True, display_progress=False, processes=1024)
except ValueError:
    print("Tissue already populated")

try:
    Detection.populate(reserve_jobs=True, display_progress=False, processes=1024)
except ValueError:
    print("Detection already populated")

try:
    Fluorescence.populate(reserve_jobs=True, display_progress=False, processes=1024)
except ValueError:
    print("Fluorescence already populated")
