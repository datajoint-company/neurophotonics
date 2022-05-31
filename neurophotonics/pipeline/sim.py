import tqdm
import numpy as np
import datajoint as dj
from .design import Design, Geometry
from .fields import ESim, DSim, EField, DField
from scipy.spatial import distance
from scipy.spatial.transform import Rotation as R
from scipy.interpolate import RegularGridInterpolator

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
                (Geometry.EPixel & key).fetch("cx", "cy", "cz"),
                (Geometry.DPixel & key).fetch("cx", "cy", "cz"),
            ]
        )

        margin = 50
        bounds_min = xyz.min(axis=-1) - margin
        bounds_max = xyz.max(axis=-1) + margin

        volume = (bounds_max - bounds_min).prod() * 1e-9  # 1e-9 is for um3 -> mm3
        npoints = int(volume * density + 0.5)

        # add one point at a time checking that it is not too close to existing points
        points = np.random.rand(1, 3) * (bounds_max - bounds_min) + bounds_min
        for i in tqdm.tqdm(range(npoints - 1)):
            while True:
                point = np.random.rand(1, 3) * (bounds_max - bounds_min) + bounds_min
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
        neuron_cross_section = 0.1  # um^2
        photons_per_joule = 1 / (2.8 * 1.6e-19)  # 2.8 eV blue photons
        cell_xyz = (Tissue & key).fetch1("cell_xyz")
        self.insert1(key)

        # iterate through each EField
        for sim_key in (ESim & (Geometry.EPixel & key)).fetch("KEY"):

            volume, pitch, *dims = (EField * ESim & key & sim_key).fetch1(
                "volume", "pitch", "volume_dimx", "volume_dimy", "volume_dimz"
            )
            volume = RegularGridInterpolator(
                (np.r_[: dims[0]], np.r_[: dims[1]], np.r_[: dims[2]]),
                volume,
                method="nearest",
                bounds_error=False,
                fill_value=0,
            )

            keys, cx, cy, cz, nx, ny, nz, tx, ty, tz = (
                Geometry.EPixel & sim_key & key
            ).fetch("KEY", "cx", "cy", "cz", "nx", "ny", "nz", "tx", "ty", "tz")

            z_basis = np.stack((nx, ny, nz)).T
            x_basis = np.stack((tx, ty, tz)).T
            y_basis = np.cross(z_basis, x_basis)

            basis = np.stack(
                (x_basis, y_basis, z_basis), axis=-1
            )  #  pixels * xyz * basis

            # assert orthonormality
            assert np.all(((basis**2).sum(axis=1) - 1) < 1e-6)
            assert np.all(np.abs((basis[:, :, 0] * basis[:, :, 1]).sum(axis=1)) < 1e-6)
            assert np.all(np.abs((basis[:, :, 1] * basis[:, :, 2]).sum(axis=1)) < 1e-6)
            assert np.all(np.abs((basis[:, :, 2] * basis[:, :, 0]).sum(axis=1)) < 1e-6)

            chunk = 2000
            for i in range(0, len(keys), chunk):
                ix = slice(i, i + chunk)

                # compute the cell coordinates in each pixels coordinates
                centers = (
                    cell_xyz[:, None, :]
                    - (np.stack((cx[ix], cy[ix], cz[ix])).T)[None, :, :]
                )  # cells x pixels x ndim
                coords = (
                    np.einsum("ijk,jkn->jin", centers, basis[ix]) / pitch
                    + np.array(dims) / 2
                )  # pixels x cells x ndim

                # emitted photons per joule
                photons = np.float32(
                    neuron_cross_section * photons_per_joule * volume(coords)
                )   # pixels x cells

                self.EPixel.insert(
                    dict(key, reemitted_photons=n, photons_per_joule=n.sum())
                    for key, n in zip(keys[ix], photons)
                )


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

        for sim_key in (DSim & (Geometry.DPixel & key)).fetch("KEY"):

            volume, pitch, *dims = (DField * DSim & key & sim_key).fetch1(
                "volume", "pitch", "volume_dimx", "volume_dimy", "volume_dimz"
            )

            volume = RegularGridInterpolator(
                (np.r_[: dims[0]], np.r_[: dims[1]], np.r_[: dims[2]]),
                volume,
                method="nearest",
                bounds_error=False,
                fill_value=0,
            )

            keys, cx, cy, cz, nx, ny, nz, tx, ty, tz = (
                Geometry.DPixel & sim_key & key
            ).fetch("KEY", "cx", "cy", "cz", "nx", "ny", "nz", "tx", "ty", "tz")

            z_basis = np.stack((nx, ny, nz)).T
            x_basis = np.stack((tx, ty, tz)).T
            y_basis = np.cross(z_basis, x_basis)

            basis = np.stack(
                (x_basis, y_basis, z_basis), axis=-1
            )  #  pixels * xyz * basis

            # assert orthonormality
            assert np.all(((basis**2).sum(axis=1) - 1) < 1e-6)
            assert np.all(np.abs((basis[:, :, 0] * basis[:, :, 1]).sum(axis=1)) < 1e-6)
            assert np.all(np.abs((basis[:, :, 1] * basis[:, :, 2]).sum(axis=1)) < 1e-6)
            assert np.all(np.abs((basis[:, :, 2] * basis[:, :, 0]).sum(axis=1)) < 1e-6)

            chunk = 2000
            for i in range(0, len(keys), chunk):
                ix = slice(i, i + chunk)
                # compute the cell position in each pixel's coordinates
                centers = (
                    cell_xyz[:, None, :]
                    - (np.stack((cx[ix], cy[ix], cz[ix])).T)[None, :, :]
                )  # cells x pixels x ndim

                # volume coordinates of all cells for all epixels
                coords = (
                    np.einsum("ijk,jkn->jin", centers, basis[ix]) / pitch
                    + np.array(dims) / 2
                )
                probabilities = volume(coords)

                self.DPixel.insert(
                    dict(
                        key,
                        detect_probabilities=probability,
                        mean_probability=probability.mean(),
                    )
                    for key, probability in zip(keys[ix], probabilities)
                )
