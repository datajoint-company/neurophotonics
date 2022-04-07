import os
import datajoint as dj
from . import demix, design, fields, sim, space


if 'custom' not in dj.config:
    dj.config['custom'] = {}

# overwrite dj.config['custom'] values with environment variables if available
dj.config['custom']['database.prefix'] = os.getenv(
    'DATABASE_PREFIX', dj.config['custom'].get('database.prefix', ''))
