import datajoint as dj
import numpy as np
from scipy.spatial.transform import Rotation as R

from .. import db_prefix
from .fields import EField, DField


schema = dj.schema(dj.config["custom"]["database.prefix"] + "phox")


@schema
class Design(dj.Lookup):
    definition = """
    design            : varchar(8)      # design code
    ---
    design_description: varchar(1000)
    """

    contents = [
        {"design": "D101", "design_description": "30 um separation. 45-degree beam"},
        {"design": "D102", "design_description": "75 um separation. 45-degree beam"},
        {"design": "D103", "design_description": "120 um separation. 45-degree beam"},
        {"design": "D104", "design_description": "120 um separation. 30-degree beam"},
        {"design": "D201", "design_description": "Steer 0. 45-degree beam"},
        {"design": "D202", "design_description": "Steer 45. 45-degree-beam"},
        {"design": "D203", "design_description": "Steer 0. 30-degree beam"},
        {"design": "D204", "design_description": "Steer 45. 30-degree-beam"},
        {
            "design": "D105",
            "design_description": "30 um separation. 45-degree beam. 50um scattering length",
        },
        {
            "design": "D106",
            "design_description": "75 um separation. 45-degree beam. 50um scattering length",
        },
        {
            "design": "D107",
            "design_description": "120 um separation. 45-degree beam. 50um scattering length",
        },
        {
            "design": "D108",
            "design_description": "120 um separation. 30-degree beam. 50um scattering length",
        },
        {
            "design": "D205",
            "design_description": "Steer 0. 45-degree beam. 50um scattering length",
        },
        {
            "design": "D206",
            "design_description": "Steer 45. 45-degree-beam. 50um scattering length",
        },
        {
            "design": "D207",
            "design_description": "Steer 0. 30-degree beam. 50um scattering length",
        },
        {
            "design": "D208",
            "design_description": "Steer 45. 30-degree-beam. 50um scattering length",
        },
        {
            "design": "D220",
            "design_description": "Steer 45 randomized directions. 45-degree-beam. 50um scattering length",
        },
        {
            "design": "D230",
            "design_description": "Steer 30 randomized directions. 30-degree-beam. 50um scattering length",
        },
    ]


@schema
class Geometry(dj.Computed):
    definition = """
    -> Design
    """

    class Shank(dj.Part):
        definition = """
        -> Geometry
        shank : smallint 
        ---
        polygon: longblob
        """

    class EPixel(dj.Part):
        definition = """  # subtable of Geometry
            -> Geometry.Shank
            epixel    : int
            ----
            -> EField
            cx: float  # um
            cy: float  # um
            cz: float  # um
            nx  : float     # norm vector (unit)
            ny  : float
            nz  : float
            tx   : float   # top vector (unit)
            ty   : float
            tz   : float
            height  : float  # um
            width   : float  # um
            thick   : float  # um
            """

    class DPixel(dj.Part):
        definition = """  # subtable of Geometry
            -> Geometry.Shank
            dpixel   : int
            ----
            -> DField
            cx : float  # um
            cy : float  # um
            cz : float  # um
            nx   : float  # norm vector (unit)
            ny   : float
            nz   : float
            tx   : float  # top vector (unit)
            ty   : float
            tz   : float
            height   : float  # um
            width    : float  # um
            thick    : float  # um
            """

    def make(self, key):
        np.random.seed(42)

        self.insert1(key)

        if key["design"] in {
            "D101",
            "D102",
            "D103",
            "D104",
            "D105",
            "D106",
            "D107",
            "D108",
        }:
            self._make_design1(key)
        elif key["design"] in {
            "D201",
            "D202",
            "D203",
            "D204",
            "D205",
            "D206",
            "D207",
            "D208",
            "D220",
            "D230",
        }:
            self._make_design2(key)
        else:
            raise NotImplementedError

    def _make_design1(self, key):
        shank_width = 120
        shank_length = 1200

        separation, esim = dict(
            D101=(30, 0),
            D102=(75, 0),
            D103=(120, 0),
            D104=(120, 1),
            D105=(30, 6),
            D106=(75, 6),
            D107=(120, 6),
            D108=(120, 7),
        )[key["design"]]

        for shank in -1, 0, 1:
            angle = np.radians(75) * shank

            rotate = R.from_euler("z", angle)
            translate = np.array(
                [
                    [
                        ((np.cos(angle) + 1) * shank_width / 2 + separation) * shank,
                        (np.sin(angle) * shank_width / 2) * shank,
                        0,
                    ]
                ]
            )
            polygon = np.float32(
                rotate.apply(
                    np.array(
                        [
                            [-shank_width / 2, 0, 0],
                            [+shank_width / 2, 0, 0],
                            [+shank_width / 2, 0, shank_length],
                            [0, 0, shank_length + shank_width / 4],
                            [-shank_width / 2, 0, shank_length],
                        ]
                    )
                )
                + translate
            )

            self.Shank().insert1(dict(key, shank=shank, polygon=polygon))

            top = rotate.apply([0, 0, 1])
            norm = rotate.apply([0, 1, 0])

            if shank in (0,):  # place D-pixels
                pixel_size = 5
                pixel_spacing = 5
                ncolumns = 22
                nrows = shank_length / pixel_spacing  # number of rows
                centers = self._make_dpixels(nrows, ncolumns)
                centers = rotate.apply(centers * pixel_spacing) + translate
                self.DPixel.insert(
                    dict(
                        key,
                        shank=shank,
                        dpixel=dpixel,
                        dsim=0,  # normal beam
                        **dict(zip(("tx", "ty", "tz"), top)),
                        **dict(zip(("nx", "ny", "nz"), norm)),
                        **dict(zip(("cx", "cy", "cz"), center)),
                        height=pixel_size,
                        width=pixel_size,
                        thick=0
                    )
                    for dpixel, center in enumerate(centers)
                )

            if shank in (-1, 1):  # place E-pixels
                pixel_size = 10
                pixel_spacing = 20
                ncolumns = 5
                nrows = shank_length / pixel_spacing
                centers = self._make_epixels(nrows, ncolumns)
                centers = rotate.apply(centers * pixel_spacing) + translate
                self.EPixel.insert(
                    dict(
                        key,
                        shank=shank,
                        epixel=epixel,
                        esim=esim,  # normal beam
                        **dict(zip(("tx", "ty", "tz"), top)),
                        **dict(zip(("nx", "ny", "nz"), norm)),
                        **dict(zip(("cx", "cy", "cz"), center)),
                        height=pixel_size,
                        width=pixel_size,
                        thick=0
                    )
                    for epixel, center in enumerate(centers)
                )

    def _make_design2(self, key):
        shank_width = 150
        shank_length = 1200
        separation = 150
        spacing = 260 if key["design"] == "D230" else 150

        for shank in range(10):
            flip = shank % 2  # rotate around the global orgin

            rotate = R.from_euler("z", flip * np.pi)
            translate = rotate.apply(
                np.array([(shank // 2 * 2 - 4.5) * spacing, -separation / 2, 0])
            )

            esim = dict(
                D201=0,
                D202=4,
                D203=1,
                D204=5,
                D205=6,
                D206=10,
                D207=7,
                D208=11,
                D220=11,
                D230=9,
            )[key["design"]]

            polygon = np.float32(
                rotate.apply(
                    np.array(
                        [
                            [-shank_width / 2, 0, 0],
                            [+shank_width / 2, 0, 0],
                            [+shank_width / 2, 0, shank_length],
                            [0, 0, shank_length + shank_width / 4],
                            [-shank_width / 2, 0, shank_length],
                        ]
                    )
                )
                + translate
            )

            self.Shank().insert1(dict(key, shank=shank, polygon=polygon))

            top = rotate.apply([0, 0, 1])
            norm = rotate.apply([0, 1, 0])

            # D-Pixels
            pixel_size = 5
            ncolumns = 22
            nrows = int(shank_length / pixel_size)  # number of rows
            centers = self._make_dpixels(nrows, ncolumns)
            centers = rotate.apply(centers * pixel_size) + translate
            self.DPixel.insert(
                dict(
                    key,
                    shank=shank,
                    dpixel=dpixel,
                    dsim=0,  # normal beam
                    **dict(zip(("tx", "ty", "tz"), top)),
                    **dict(zip(("nx", "ny", "nz"), norm)),
                    **dict(zip(("cx", "cy", "cz"), center)),
                    height=pixel_size,
                    width=pixel_size,
                    thick=0
                )
                for dpixel, center in enumerate(centers)
            )

            # E-Pixels
            pixel_size = 10
            pixel_spacing = 20
            ncolumns = 5
            nrows = int(shank_length / pixel_spacing)
            centers = self._make_epixels(nrows, ncolumns)
            centers = rotate.apply(centers * pixel_spacing) + translate
            if key["design"] not in ["D220", "D230"]:
                checkerboard = self._make_checkerboard(nrows, ncolumns)
                if shank < 2:  # two side shanks angled only one way
                    checkerboard = np.ones_like(checkerboard)
                tops = checkerboard[:, None] * top
            else:
                tops = np.array(
                    [
                        (np.random.randint(2) * 2 - 1)
                        * (np.cross(top, norm) if np.random.random() < 1 / 3 else top)
                        for _ in range(nrows * ncolumns)
                    ]
                )

            self.EPixel.insert(
                dict(
                    key,
                    shank=shank,
                    epixel=epixel,
                    esim=esim,
                    **dict(zip(("tx", "ty", "tz"), _top)),
                    **dict(zip(("nx", "ny", "nz"), norm)),
                    **dict(zip(("cx", "cy", "cz"), center)),
                    height=pixel_size,
                    width=pixel_size,
                    thick=0
                )
                for epixel, (center, _top) in enumerate(zip(centers, tops))
            )

    @staticmethod
    def _make_epixels(nrows, ncolumns):
        return make_grid(np.r_[-ncolumns / 2 + 0.5 : ncolumns / 2], np.r_[0.5:nrows])

    @staticmethod
    def _make_checkerboard(nrows, ncolumns):
        return (
            np.array(
                [g.flatten() for g in np.meshgrid(np.r_[:nrows], np.r_[:ncolumns])]
            ).sum(axis=0)
            % 2
        ) * 2 - 1

    @staticmethod
    def _make_dpixels(nrows, ncolumns):
        return np.vstack(
            [
                # solid row 0
                make_grid(
                    np.r_[-ncolumns / 2 + 0.5 : ncolumns / 2], np.r_[0.5:nrows:4]
                ),
                # gapped row 1, odds and evens
                make_grid(
                    np.r_[-ncolumns / 2 + 0.5 : ncolumns / 2 : 4], np.r_[1.5:nrows:4]
                ),
                make_grid(
                    np.r_[-ncolumns / 2 + 1.5 : ncolumns / 2 : 4], np.r_[1.5:nrows:4]
                ),
                # gapped row 2, odds and evens
                make_grid(
                    np.r_[-ncolumns / 2 + 0.5 : ncolumns / 2 : 4], np.r_[2.5:nrows:4]
                ),
                make_grid(
                    np.r_[-ncolumns / 2 + 1.5 : ncolumns / 2 : 4], np.r_[2.5:nrows:4]
                ),
                # solid row 3
                make_grid(
                    np.r_[-ncolumns / 2 + 0.5 : ncolumns / 2], np.r_[3.5:nrows:4]
                ),
            ]
        )


def make_grid(row, column):
    """
    make a grid of centroids
    """
    return np.array(
        [
            g.flatten()
            for g in np.meshgrid(
                row,
                [0],
                column,
            )
        ]
    ).T
