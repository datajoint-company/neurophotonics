from neurophotonics.design import Geometry

Geometry.populate(reserve_jobs=True, display_progress=False, processes=1024)


print(
    "keys processed:",
    Geometry.progress(display=False)[0],
)
