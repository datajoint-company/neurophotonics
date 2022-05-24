from neurophotonics.fields import *

EField.populate(reserve_jobs=True, processes=1024, display_progress=False)
DField.populate(reserve_jobs=True, processes=1024, display_progress=False)

print(
    "keys processed:",
    EField.progress(display=False)[0] + DField.progress(display=False)[0],
)
