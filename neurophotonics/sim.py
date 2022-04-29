import tqdm
import numpy as np
import datajoint as dj
from .design import Geometry
from .fields import ESim, DSim, EField, DField
from scipy.spatial import distance
from scipy.spatial.transform import Rotation as R

schema = dj.schema(dj.config["custom"]["database.prefix"] + "photonics")
schema.spawn_missing_classes()


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

        def expand_over_shanks():
            xyz = np.hstack(
                [
                    (Geometry.Emitter & key).fetch("e_center_x", "e_center_y", "e_center_z"),
                    (Geometry.Detector & key).fetch("d_center_x", "d_center_y", "d_center_z"),
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
                    point = np.random.rand(1, 3) * (bounds_max - bounds_min) + bounds_min
                    if distance.cdist(points, point).min() > min_distance:
                        break
                points = np.vstack((points, point))

            return volume, margin, npoints, points

        # Work in progress
        def in_front_of_shanks():
            margin = 50  # Not using this guy, it can be nullable?
            h = 150
            w = 150
            l = 1200
            volume = h * w * l * 1e-9
            npoints = int(volume * density)
            print(npoints)

            points = np.random.rand(1, 3) * h / 2
            while abs(points.shape[0] / volume - density) < density * 1.50:
                point = np.random.rand(1, 3) * h / 2
                if distance.cdist(points, point).min() > min_distance:
                    points = np.vstack((points, point))

            for i in tqdm.tqdm(range(npoints - 1)):
                while True:
                    point = np.random.rand(1, 3) * h / 2
                    if distance.cdist(points, point).min() > min_distance:
                        break
                points = np.vstack((points, point))

            # Rotate the tissue by 45 degrees along the z axis.
            r = R.from_euler("z", 45, degrees=True)
            points = r.apply(points)

            # Translate them to the front of the detector shank.
            points += [0, 166, 0]

            return volume, margin, npoints, points

        volume, margin, npoints, points = expand_over_shanks()

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

    class Emitter(dj.Part):
        definition = """
        # Fluorescence produced by cells per Joule of illumination
        -> master
        -> Geometry.Emitter
        ---
        reemitted_photons  : longblob   # photons emitted from cells per joule of illumination
        photons_per_joule : float  # total photons from all cells
        """

    def make(self, key):
        neuron_cross_section = 1e-4  # um^2
        photons_per_joule = 1 / (2.8 * 1.6e-19)  # 2.8 eV blue photons
        cell_xyz = (Tissue & key).fetch1("cell_xyz")
        self.insert1(key)
        for esim_key in (ESim & (Geometry.Emitter & key)).fetch("KEY"):
            volume, pitch, *dims = (EField * ESim & esim_key).fetch1(
                "volume", "pitch", "volume_dimx", "volume_dimy", "volume_dimz"
            )
            dims = np.array(dims)
            entries = []
            for (
                emit_key,
                e_center_x,
                e_center_y,
                e_center_z,
                e_norm_x,
                e_norm_y,
                e_norm_z,
                e_top_x,
                e_top_y,
                e_top_z,
            ) in tqdm.tqdm(
                zip(
                    *(Geometry.Emitter & key & esim_key).fetch(
                        "KEY",
                        "e_center_x",
                        "e_center_y",
                        "e_center_z",
                        "e_norm_x",
                        "e_norm_y",
                        "e_norm_z",
                        "e_top_x",
                        "e_top_y",
                        "e_top_z",
                    )
                )
            ):
                # cell positions in volume coordinates
                e_xyz = e_center_x, e_center_y, e_center_z
                z_basis = np.array([e_norm_x, e_norm_y, e_norm_z])
                y_basis = np.array([e_top_x, e_top_y, e_top_z])
                x_basis = np.cross(z_basis, y_basis)
                assert abs(x_basis @ y_basis) < 1e-4
                assert abs(x_basis @ z_basis) < 1e-4
                assert abs(y_basis @ z_basis) < 1e-4
                assert abs(x_basis @ x_basis - 1) < 1e-4
                assert abs(y_basis @ y_basis - 1) < 1e-4
                assert abs(z_basis @ z_basis - 1) < 1e-4

                vxyz = np.int16(
                    np.round(
                        (cell_xyz - e_xyz) @ np.vstack((x_basis, y_basis, z_basis)).T / pitch
                        + dims / 2
                    )
                )

                # photon counts
                v = (
                    neuron_cross_section
                    * photons_per_joule
                    * np.array(
                        [
                            volume[q[0], q[1], q[2]]
                            if 0 <= q[0] < dims[0] and 0 <= q[1] < dims[1] and 0 <= q[2] < dims[2]
                            else 0
                            for q in vxyz
                        ]
                    )
                )
                entries.append(
                    dict(
                        key,
                        **emit_key,
                        reemitted_photons=np.float32(v),
                        photons_per_joule=v.sum()
                    )
                )
            self.Emitter().insert(entries)


@schema
class Detection(dj.Computed):
    definition = """
    -> Tissue
    """

    class Detector(dj.Part):
        definition = """
        # Fraction of photons detected from each cell per detector
        -> master
        -> Geometry.Detector
        ---
        detect_probabilities  : longblob   # fraction of photons detected from each neuron
        mean_probability : float  # mean probability of detection across all neurons
        """

    def make(self, key):
        cell_xyz = (Tissue & key).fetch1("cell_xyz")
        self.insert1(key)
        for dsim_key in (DSim & (Geometry.Detector & key)).fetch("KEY"):
            volume, pitch, *dims = (DField * DSim & dsim_key).fetch1(
                "volume", "pitch", "volume_dimx", "volume_dimy", "volume_dimz"
            )
            volume = (
                0.5 * volume / volume.max()
            )  # just in case. Max detection should already be ~0.5. Update after additional sim verifications

            dims = np.array(dims)

            entries = []
            for (
                detect_key,
                d_center_x,
                d_center_y,
                d_center_z,
                d_norm_x,
                d_norm_y,
                d_norm_z,
                d_top_x,
                d_top_y,
                d_top_z,
            ) in tqdm.tqdm(
                zip(
                    *(Geometry.Detector & key & dsim_key).fetch(
                        "KEY",
                        "d_center_x",
                        "d_center_y",
                        "d_center_z",
                        "d_norm_x",
                        "d_norm_y",
                        "d_norm_z",
                        "d_top_x",
                        "d_top_y",
                        "d_top_z",
                    )
                )
            ):
                # cell positions in volume coordinates
                d_xyz = d_center_x, d_center_y, d_center_z
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
                        (cell_xyz - d_xyz) @ np.vstack((x_basis, y_basis, z_basis)).T / pitch
                        + dims / 2
                    )
                )
                # photon counts
                v = np.array(
                    [
                        volume[q[0], q[1], q[2]]
                        if 0 <= q[0] < dims[0] and 0 <= q[1] < dims[1] and 0 <= q[2] < dims[2]
                        else 0
                        for q in vxyz
                    ]
                )
                entries.append(
                    dict(
                        key,
                        **detect_key,
                        detect_probabilities=np.float32(v),
                        mean_probability=v.sum()
                    )
                )
            self.Detector().insert(entries)
