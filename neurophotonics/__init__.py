import os
import datajoint as dj

if 'custom' not in dj.config:
    dj.config['custom'] = {}

# overwrite dj.config['custom'] values with environment variables if available
dj.config['custom']['database.prefix'] = os.getenv(
    'DATABASE_PREFIX', dj.config['custom'].get('database.prefix', ''))

from . import space, fields, design, sim, demix

