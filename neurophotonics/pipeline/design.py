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
        {
            "design": "D101",
            "design_description": "30 um separation. 45-degree beam",
        },
        {
            "design": "D102",
            "design_description": "75 um separation. 45-degree beam",
        },
        {
            "design": "D103",
            "design_description": "120 um separation. 45-degree beam",
        },
        {
            "design": "D104",
            "design_description": "120 um separation. 30-degree beam",
        },
        {
            "design": "D201",
            "design_description": "Steer 0. 45-degree beam"
        },
        {
            "design": "D202",
            "design_description": "Steer 45. 45-degree-beam",
        },
        {
            "design": "D201",
            "design_description": "Steer 0. 30-degree beam",
        },
        {
            "design": "D202",
            "design_description": "Steer 45. 30-degree-beam",
        },
    ]


@schema
class Geometry(dj.Computed):
    definition = """
    -> Design
    """

    class Shank(dj.Part):
        definition =  """
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
        self.insert1(key)

        if key['design'] in {"D101", "D102", "D103", "D104"}:       # Designs 1
            
            shank_width = 120
            shank_length = 2500
            
            angle = np.radians(75)
            separation = {"D101": 30,  "D102": 75, "D103": 120, "D104": 120}[key['design']]
            
            for shank in -1, 0, 1:
                translate = np.array([[
                    (np.cos(angle) * shank_width/2 + separation) * shank,  
                    (np.sin(angle) * shank_width/2),
                    0
                ]])
                rotate = R.from_euler('z', angle * shank)
                polygon = np.float32(rotate.apply(
                    np.array(
                    [
                        [-shank_width/2, 0, 0], 
                        [+shank_width/2, 0, 0],
                        [+shank_width/2, 0, shank_length],
                        [0, 0, shank_length + shank_width/4],
                        [-shank_width/2, 0, shank_length],
                        ])) + translate)
                self.Shank().insert1(dict(key, shank=shank, polygon=polygon))
                
                if shank in (-1, 1):   # place D-pixels

                    top = rotate.apply([0, 0, 1])
                    norm = rotate.apply([0, 1, 0])

                    # define emitter centroids
                    pixel_size = 5
                    ncolumns = 22  
                    nrows = shank_length / pixel_size  # number of rows
                    rows = np.vstack([
                        # solid row 0 
                        np.array([g.flatten() for g in np.meshgrid(
                            np.r_[-ncolumns/2 + .5: ncolumns/2 ], [0], np.r_[0.5:nrows:4])]).T, 
                        # gapped row 1, odds and evens 
                        np.array([g.flatten() for g in np.meshgrid(
                            np.r_[-ncolumns/2 + .5: ncolumns/2 :4], [0], np.r_[1.5:nrows:4])]).T, 
                        np.array([g.flatten() for g in np.meshgrid(
                            np.r_[-ncolumns/2 + 1.5: ncolumns/2 :4], [0], np.r_[1.5:nrows:4])]).T,
                        # gapped row 2, odds and evens 
                        np.array([g.flatten() for g in np.meshgrid(
                            np.r_[-ncolumns/2 + .5: ncolumns/2 :4], [0], np.r_[2.5:nrows:4])]).T, 
                        np.array([g.flatten() for g in np.meshgrid(
                            np.r_[-ncolumns/2 + 1.5: ncolumns/2 :4], [0], np.r_[2.5:nrows:4])]).T,
                        # solid row 3
                        np.array([g.flatten() for g in np.meshgrid(
                            np.r_[-ncolumns/2 + .5: ncolumns/2 ], [0], np.r_[3.5:nrows:4])]).T])

                    rows = rotate.apply(rows * pixel_size) + translate

                    self.DPixel.insert(
                        dict(
                            key, 
                            shank=shank, 
                            dpixel=dpixel,
                            dsim=0, # normal beam
                            **dict(zip(('tx', 'ty', 'tz'), top)),
                            **dict(zip(('nx', 'ny', 'nz'), norm)),
                            **dict(zip(('cx', 'cy', 'cz'), row)),
                            height=pixel_size,
                            width=pixel_size,
                            thick=0) for dpixel, row in enumerate(rows))

                if shank == 0:   # place E-pixels
                    pixel_size = 10
                    ncolumns = 5
                    nrows = shank_length / pixel_size / 2
                    rows = np.array([g.flatten() for g in np.meshgrid(
                        np.r_[-ncolumns/2+.5:ncolumns/2], [0], np.r_[:nrows] * 2 + 1)]).T 
                    rows = rotate.apply(rows * pixel_size) + translate

                    self.EPixel.insert(
                        dict(
                            key, 
                            shank=shank, 
                            epixel=epixel,
                            esim=1 if key['design'] == "D104" else 0, # normal beam
                            **dict(zip(('tx', 'ty', 'tz'), top)),
                            **dict(zip(('nx', 'ny', 'nz'), norm)),
                            **dict(zip(('cx', 'cy', 'cz'), row)),
                            height=pixel_size,
                            width=pixel_size,
                            thick=0) for epixel, row in enumerate(rows))