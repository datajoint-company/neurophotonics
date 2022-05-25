import tqdm
import numpy as np
import datajoint as dj
from .design import Design, Geometry
from .fields import ESim, DSim, EField, DField
from scipy.spatial import distance
from scipy.spatial.transform import Rotation as R
from multiprocess import Pool
from multiprocess import cpu_count
import gc

from .. import db_prefix

from .design import Geometry

schema = dj.schema(db_prefix + "phox")


@schema
class Tissue(dj.Computed):
    definition = """
    # Point sources of fluorescence
    -> Geometry
    ---
    volume       : float     # (mm^3)
    margin       : float     # (um) margin to include on boundaries
    npoints      : int       # total number of points in volume
    min_distance : float     # (um)
    cell_xyz     : longblob  # cell
    """

    def make(self, key):
        density = 120000
        min_distance = 8.0

        xyz = np.hstack(
            [
                (Geometry.EPixel & key).fetch(
                    "cx", "cy", "cz"
                ),
                (Geometry.DPixel & key).fetch(
                    "cx", "cy", "cz"
                ),
            ]
        )

        margin = 50
        bounds_min = xyz.min(axis=-1) - margin
        bounds_max = xyz.max(axis=-1) + margin

        volume = (bounds_max - bounds_min).prod() * 1e-9  # 1e-9 is for um3 -> mm3
        npoints = int(volume * density + 0.5)

        points = np.random.rand(1, 3) * (bounds_max - bounds_min) + bounds_min
        for i in tqdm.tqdm(range(npoints - 1)):
            while True:
                point = (
                    np.random.rand(1, 3) * (bounds_max - bounds_min) + bounds_min
                )
                if distance.cdist(points, point).min() > min_distance:
                    break
            points = np.vstack((points, point))

        self.insert1(
            dict(
                key,
                volume=volume,
                margin=margin,
                npoints=npoints,
                min_distance=min_distance,
                cell_xyz=points,
            )
        )


@schema
class Fluorescence(dj.Computed):
    definition = """
    -> Tissue
    """

    class EPixel(dj.Part):
        definition = """
        # Fluorescence produced by cells per Joule of illumination
        -> master
        -> Geometry.EPixel
        ---
        reemitted_photons  : longblob   # photons emitted from cells per joule of illumination
        photons_per_joule : float  # total photons from all cells
        """

    def make(self, key):
        neuron_cross_section = 1e-4  # um^2
        photons_per_joule = 1 / (2.8 * 1.6e-19)  # 2.8 eV blue photons
        cell_xyz = (Tissue & key).fetch1("cell_xyz")
        self.insert1(key)

        # iterate through each EField
        for esim_key in (ESim & (Geometry.EPixel & key)).fetch('KEY'):
            volume, pitch, *dims = (EField * ESim & esim_key).fetch1(
                'volume', 'pitch', 'volume_dimx', 'volume_dimy', 'volume_dimz')
            dims = np.array(dims)

            keys, cx, cy, cz, nx, ny, nz, tx, ty, tz = (Geometry.EPixel & ekey).fetch1(
                    'KEY', 'cx', 'cy', 'cz', 'nx', 'ny', 'nz', 'tx', 'ty', 'tz')
            centers = np.hstack((cx, cy, cz))
            norms = np.hstack((nx, ny, nz))
            tops = np.hpstack((tx, ty, tz))
            
            # iterate through E-Fields
            for ekey in tqdm.tqdm((Geometry.Emitter & key & esim_key).fetch('KEY')):
                cx, cy, cz, nx, ny, nz, tx, ty, tz = (Geometry.EPixel & ekey).fetch1(
                    'cx', 'cy', 'cz', 'nx', 'ny', 'nz', 'tx', 'ty', 'tz')
                center = np.array(('cx', 'cy', 'cz'))
                z_basis = np.array((nx, ny, nz))
                y_basis = np.array((tx, ty, tz))
                x_basis = np.cross(z_basis, y_basis)
                assert abs(x_basis @ y_basis) < 1e-4
                assert abs(x_basis @ z_basis) < 1e-4
                assert abs(y_basis @ z_basis) < 1e-4
                assert abs(x_basis @ x_basis - 1) < 1e-4
                assert abs(y_basis @ y_basis - 1) < 1e-4
                assert abs(z_basis @ z_basis - 1) < 1e-4

                vcenters = np.int16(np.round(
                    (cell_xyz - center) @ np.vstack((x_basis, y_basis, z_basis)).T / pitch + dims/2))

                # photon counts
                v = neuron_cross_section * photons_per_joule * np.array([
                    volume[q[0], q[1], q[2]] if 
                    0 <= q[0] < dims[0] and 
                    0 <= q[1] < dims[1] and 
                    0 <= q[2] < dims[2] else 0 for q in vcenters])

                self.EPixel().insert1(
                    dict(key, **ekey, reemitted_photons=np.float32(v), photons_per_joule=v.sum()))

        gc.collect()


@schema
class Detection(dj.Computed):
    definition = """
    -> Tissue
    """

    class DPixel(dj.Part):
        definition = """
        # Fraction of photons detected from each cell per detector
        -> master
        -> Geometry.DPixel
        ---
        detect_probabilities  : longblob   # fraction of photons detected from each neuron
        mean_probability : float  # mean probability of detection across all neurons
        """

    def make(self, key):
        cell_xyz = (Tissue & key).fetch1("cell_xyz")
        self.insert1(key)
        volume = (DField & key).fetch1("volume")

        pitch, volume_dimx, volume_dimy, volume_dimz = (DSim & key).fetch1(
            "pitch", "volume_dimx", "volume_dimy", "volume_dimz")

        cx, cy, cz, nx, ny, nz, tx, ty, tz = (Geometry.DPixel & key).fetch(
            'cx', 'cy', 'cz', 'nx', 'ny', 'nz', 'tx', 'ty', 'tz')
        
        dims = np.array([volume_dimx, volume_dimy, volume_dimz])

        # cell positions in volume coordinates
        z_basis = np.array([d_norm_x, d_norm_y, d_norm_z])
        y_basis = np.array([d_top_x, d_top_y, d_top_z])
        x_basis = np.cross(z_basis, y_basis)
        assert abs(x_basis @ y_basis) < 1e-4
        assert abs(x_basis @ z_basis) < 1e-4
        assert abs(y_basis @ z_basis) < 1e-4
        assert abs(x_basis @ x_basis - 1) < 1e-4
        assert abs(y_basis @ y_basis - 1) < 1e-4
        assert abs(z_basis @ z_basis - 1) < 1e-4
        vxyz = np.int16(
            np.round(
                (cell_xyz - d_xyz)
                @ np.vstack((x_basis, y_basis, z_basis)).T
                / pitch
                + dims / 2
            )
        )
        # photon counts
        v = np.array(
            [
                volume[q[0], q[1], q[2]]
                if 0 <= q[0] < dims[0]
                and 0 <= q[1] < dims[1]
                and 0 <= q[2] < dims[2]
                else 0
                for q in vxyz
            ]
        )
        entry = dict(
            key,
            **detect_key,
            detect_probabilities=np.float32(v),
            mean_probability=v.sum()
        )
        Detection.DPixel.insert1(entry, ignore_extra_fields=True)

        try:
            with Pool(cpu_count()) as p:
                p.starmap(calculate, tqdm.tqdm(input_pars, total=len(input_pars)))
        except Exception as e:
            print(e)
            with dj.config(safemode=False):
                (self & key).delete()

        gc.collect()
