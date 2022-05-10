from neurophotonics.sim import Tissue, Fluorescence, Detection
from scripts.helper import keys_used

print(
    "keys processed:",
    keys_used(
        Tissue.populate(reserve_jobs=True, display_progress=False, processes=1024)
    )
    + keys_used(
        Detection.populate(reserve_jobs=True, display_progress=False, processes=2)
    )
    + keys_used(
        Fluorescence.populate(reserve_jobs=True, display_progress=False, processes=2)
    ),
)
