import logging
from datajoint_utilities.dj_worker import DataJointWorker, WorkerLog, ErrorLog

from neurophotonics import db_prefix
from neurophotonics.pipeline import design, fields, sim, demix

_logger = logging.getLogger(__name__)
_logger.setLevel('INFO')

__all__ = ['standard_worker', 'WorkerLog', 'ErrorLog']


# -------- Define process(s) --------
org_name, workflow_name, _ = db_prefix.split('_')

worker_db_prefix = f'{org_name}_support_{workflow_name}_'
worker_schema_name = worker_db_prefix + "workerlog"
autoclear_error_patterns = []

# all processing steps for this whole pipeline
standard_worker = DataJointWorker('standard_worker',
                                  worker_schema_name,
                                  db_prefix=[db_prefix, worker_db_prefix],
                                  run_duration=1,
                                  sleep_duration=30,
                                  autoclear_error_patterns=autoclear_error_patterns)

standard_worker(fields.EField, processes=1024)
standard_worker(fields.DField, processes=1024)

standard_worker(design.Geometry, processes=1024)

standard_worker(sim.Tissue, processes=1024)
standard_worker(sim.Detection, processes=2)
standard_worker(sim.Fluorescence, processes=2)

standard_worker(demix.IlluminationCycle, processes=2)
standard_worker(demix.Demix, processes=1024)
standard_worker(demix.Cosine, processes=1)
standard_worker(demix.SpikeSNR, processes=1)
