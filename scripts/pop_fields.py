from neurophotonics.fields import *
from scripts.helper import keys_used

print(
    "keys processed:",
    keys_used(
        EField.populate(reserve_jobs=True, processes=1024, display_progress=False)
    )
    + keys_used(
        DField.populate(reserve_jobs=True, processes=1024, display_progress=False)
    ),
)
