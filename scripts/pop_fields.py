from neurophotonics.fields import *


EField.populate(reserve_jobs=True, processes=1024, display_progress=False)
DField.populate(reserve_jobs=True, processes=1024, display_progress=False)
# try:
#    EField.populate(reserve_jobs=True, processes = 1024, display_progress=False)
# except ValueError:
#    print('EField already populated')
#
# try:
#    DField.populate(reserve_jobs=True, processes = 1024, display_progress=False)
# except ValueError:
#    print('DField already populated')
