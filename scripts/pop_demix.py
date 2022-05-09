from neurophotonics.demix import *
from scripts.helper import keys_used

print(
    "keys processed:",
    keys_used(
        IlluminationCycle.populate(
            reserve_jobs=True, processes=1024, display_progress=False
        )
    )
    + keys_used(
        Demix.populate(display_progress=False, processes=1024, reserve_jobs=True)
    )
    + keys_used(
        Cosine.populate(reserve_jobs=True, display_progress=False, processes=1024)
    )
    + keys_used(
        SpikeSNR.populate(reserve_jobs=True, display_progress=False, processes=1024)
    ),
)
