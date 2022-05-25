import numpy as np
import datajoint as dj
from .space import Space
from matplotlib import pyplot as plt

from .. import db_prefix


schema = dj.schema(db_prefix + "phox")


@schema
class DSim(dj.Lookup):
    definition = """
    # Detector Field Specification
    dsim : int
    ---
    dsim_description            : varchar(1024)
    detector_type = 'one-sided' : varchar(30)  # choice in simulation
    detector_width = 10.00      : decimal(5,2) # (um) along x-axis
    detector_height = 10.00     : decimal(5,2) # (um) along y-axis
    anisotropy = 0.88           : float        # factor in the Henyey-Greenstein formula
    absorption_length = 14000   : float        # (um) average travel path before an absorption event
    scatter_length = 100        : float        # (um) average travel path before a scatter event
    volume_dimx = 1000          : int unsigned # (voxels)
    volume_dimy = 1000          : int unsigned # (voxels)
    volume_dimz = 1000          : int unsigned # (voxels)
    pitch = 2.2                 : float        # (um)  spatial sampling period of the model volume
    """

    contents = [
        dict(
            dsim=0,
            dsim_description="5x5 lambertian",
            detector_width=5.00,
            detector_height=5.00,
            anisotropy=0.88,
            absorption_length=14000,
            scatter_length=100,
            volume_dimx=1000,
            volume_dimy=1000,
            volume_dimz=1000,
            pitch=2.2,
        ),
    ]


@schema
class DField(dj.Computed):
    definition = """
    # Detector Field Reference Volume
    -> DSim
    ---
    volume        : blob@photonics # probability of a photon emitted at given point getting picked up by the given detector
    max_value     : float          # should be < 1.0
    total_photons : int unsigned
    """

    def compute(self, key):
        spec = (DSim & key).fetch1()

        kwargs = {
            k: spec[k]
            for k in spec
            if k
            in {
                "pitch",
                "anisotropy",
                "scatter_length",
                "absorption_length",
                "detector_type",
            }
        }

        kwargs.update(
            dims=tuple(spec[k] for k in ("volume_dimx", "volume_dimy", "volume_dimz")),
            emitter_spread="spherical",
            emitter_size=(
                float(spec["detector_width"]),
                float(spec["detector_height"]),
                0,
            ),
        )

        space = Space(**kwargs)
        space.run(hops=1_000_000)
        space.volume *= space.emitter_area
        return space

    def make(self, key):
        space = self.compute(key)
        self.insert1(
            dict(
                key,
                volume=np.float32(space.volume),
                max_value=space.volume.max(),
                total_photons=space.total_count,
            )
        )

    def plot(self, axis=None, gamma=0.7, cmap="gray_r", title=""):
        from matplotlib_scalebar.scalebar import ScaleBar

        info = (self * DSim).fetch1()
        if axis is None:
            _, axis = plt.subplots(1, 1, figsize=(8, 8))
        axis.imshow((info["volume"].sum(axis=0)) ** gamma, cmap=cmap)
        axis.axis(False)
        scale_bar = ScaleBar(info["pitch"] * 1e-6)
        axis.add_artist(scale_bar)
        title = f"{title}\n{info['total_photons'] / 1e6:0.2f} million simulated photons"
        axis.set_title(title)


@schema
class ESim(dj.Lookup):
    definition = """
    # Emission Field Specification
    esim : int
    ---
    esim_description          : varchar(1024)
    beam_compression          : float
    y_steer                   : float         # the steer angle in the plane of the shank
    emitter_width = 10.00     : decimal(5,2)  # (um) along x-axis
    emitter_height = 10.00    : decimal(5,2)  # (um) along y-axis
    anisotropy = 0.88         : float         # factor in the Henyey-Greenstein formula
    absorption_length = 14000 : float         # (um) average travel path before a absorption event
    scatter_length = 100      : float         # (um) average travel path before a scatter event
    volume_dimx = 1000        : int unsigned  # (voxels)
    volume_dimy = 1000        : int unsigned  # (voxels)
    volume_dimz = 1000        : int unsigned  # (voxels)
    beam_xy_aspect = 1.0      : float         # compression of y. E.g. 2.0 means that y is compressed by factor of 2
    pitch = 2.2               : float         # (um) spatial sampling period of the model volume
    """

    contents = [
        dict(
            esim=0,
            esim_description="Narrowed to pi/4, steer 0",
            beam_compression=0.25,
            y_steer=0.0,
            emitter_width=10.0,
            emitter_height=10.00,
            anisotropy=0.88,
            absorption_length=14000.0,
            scatter_length=100.0,
            volume_dimx=1000,
            volume_dimy=1000,
            volume_dimz=1000,
            beam_xy_aspect=1.0,
            pitch=2.2,
        ),
        dict(
            esim=1,
            esim_description="Narrow pi/6, steer 0",
            beam_compression=0.167,
            y_steer=0.0,
            emitter_width=10.0,
            emitter_height=10.00,
            anisotropy=0.88,
            absorption_length=14000.0,
            scatter_length=100.0,
            volume_dimx=1000,
            volume_dimy=1000,
            volume_dimz=1000,
            beam_xy_aspect=1.0,
            pitch=2.2,
        ),
        dict(
            esim=2,
            esim_description="Narrow to pi/4, steer 30",
            beam_compression=0.25,
            y_steer=np.pi / 6,
            emitter_width=10.0,
            emitter_height=10.00,
            anisotropy=0.88,
            absorption_length=14000.0,
            scatter_length=100.0,
            volume_dimx=1000,
            volume_dimy=1000,
            volume_dimz=1000,
            beam_xy_aspect=1.0,
            pitch=2.2,
        ),
        dict(
            esim=3,
            esim_description="Narrowed to pi/6, steer 30",
            beam_compression=0.167,
            y_steer=np.pi / 6,
            emitter_width=10.0,
            emitter_height=10.00,
            anisotropy=0.88,
            absorption_length=14000.0,
            scatter_length=100.0,
            volume_dimx=1000,
            volume_dimy=1000,
            volume_dimz=1000,
            beam_xy_aspect=1.0,
            pitch=2.2,
        ),
        dict(
            esim=4,
            esim_description="Narrow to pi/4, steer 45",
            beam_compression=0.25,
            y_steer=np.pi / 4,
            emitter_width=10.0,
            emitter_height=10.00,
            anisotropy=0.88,
            absorption_length=14000.0,
            scatter_length=100.0,
            volume_dimx=1000,
            volume_dimy=1000,
            volume_dimz=1000,
            beam_xy_aspect=1.0,
            pitch=2.2,
        ),
        dict(
            esim=5,
            esim_description="Narrowed to pi/6, steer 45",
            beam_compression=0.167,
            y_steer=np.pi / 4,
            emitter_width=10.0,
            emitter_height=10.00,
            anisotropy=0.88,
            absorption_length=14000.0,
            scatter_length=100.0,
            volume_dimx=1000,
            volume_dimy=1000,
            volume_dimz=1000,
            beam_xy_aspect=1.0,
            pitch=2.2,
        ),
    ]


@schema
class EField(dj.Computed):
    definition = """
    # Emitter Field Reference Volume
    -> ESim
    ---
    volume        : blob@photonics # probability of a photon emitted at given point getting picked up by the given detector
    total_photons : int unsigned
    """

    def compute(self, key, hops=1_000_000):

        spec = (ESim & key).fetch1()

        # pass arguments from lookup to function
        kwargs = {
            k: spec[k]
            for k in spec
            if k
            in {
                "pitch",
                "anisotropy",
                "scatter_length",
                "y_steer",
                "beam_compression",
                "beam_xy_aspect",
                "absorption_length",
            }
        }

        kwargs.update(
            dims=tuple(spec[k] for k in ("volume_dimx", "volume_dimy", "volume_dimz")),
            emitter_size=(
                float(spec["emitter_width"]),
                float(spec["emitter_height"]),
                0,
            ),
        )
        space = Space(**kwargs)
        space.run(hops=hops)
        return space

    def make(self, key):
        space = self.compute(key)
        self.insert1(
            dict(key, volume=np.float32(space.volume), total_photons=space.total_count)
        )

    def plot(self, figsize=(8, 8), axis=None, gamma=0.7, cmap="magma", title=""):
        from matplotlib_scalebar.scalebar import ScaleBar

        info = (self * ESim).fetch1()
        if axis is None:
            _, axis = plt.subplots(1, 1, figsize=(8, 8))
        axis.imshow((info["volume"].sum(axis=0)) ** gamma, cmap=cmap)
        axis.axis(False)
        scale_bar = ScaleBar(info["pitch"] * 1e-6)
        axis.add_artist(scale_bar)
        title = f"{title}\n{info['total_photons'] / 1e6:0.2f} million simulated photons"
        axis.set_title(title)
