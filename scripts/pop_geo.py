from neurophotonics.design import Geometry
from scripts.helper import keys_used

print(
    "keys processed:",
    keys_used(
        Geometry.populate(reserve_jobs=True, display_progress=False, processes=1024)
    ),
)
