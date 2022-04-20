from neurophotonics.fields import ESim, DSim
from neurophotonics.design import Design
import numpy as np


def ingest_ESimDSim():
    DSim.insert1(
        dict(
            dsim=0,
            detector_type="one-sided",
            detector_height=5.00,
            detector_width=5.00,
            anisotropy=0.88,
            scatter_length=100,
            absorption_length=14000,
            volume_dimx=1000,
            volume_dimy=1000,
            volume_dimz=1000,
            pitch=2.2,
            dsim_description="100% Efficient Lambertian 5x5 rect",
        ),
        skip_duplicates=True,
    )

    ESim.insert1(
        dict(
            esim=0,
            esim_description="Narrowed to pi/4, steered 0",
            beam_compression=1 / 4,
            y_steer=0 * np.pi,
            emitter_width=10.00,
            emitter_height=10.00,
            anisotropy=0.88,
            absorption_length=14000,
            scatter_length=100,
            volume_dimx=1000,
            volume_dimy=1000,
            volume_dimz=1000,
            beam_xy_aspect=1.0,
        ),
        skip_duplicates=True,
    )


def ingest_design():
    Design.insert1(
        dict(
            design=1,
            design_title="Design1",
            design_description="1 detector, 2-emitters, 153 unit apart, 45 degrees",
            design_path="./",
            geometry_file="Design1.csv",
            center_offset=(0, 0, 0),
            efields=0,
            dfields=0,
        ),
        skip_duplicates=True,
    )
