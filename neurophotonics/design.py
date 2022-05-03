import pandas as pd
import datajoint as dj
from .fields import EField, DField


schema = dj.schema(dj.config["custom"]["database.prefix"] + "photonics")


@schema
class Design(dj.Lookup):
    definition = """
    design            : smallint      # design number
    ---
    design_title      : varchar(255)
    design_description: varchar(1000)
    design_path       : varchar(255)
    geometry_file     : varchar(255)
    center_offset     : blob          # offset from legacy implementation
    efields           : blob          # efield selection
    dfields           : blob          # dfield selection
    """

    contents = []


@schema
class Geometry(dj.Imported):
    definition = """
    -> Design
    ---
    """

    class Emitter(dj.Part):
        definition = """  # subtable of Geometry
            -> master
            emitter    :int
            ----
            -> EField
            e_center_x: float  # um
            e_center_y: float  # um
            e_center_z: float  # um
            e_norm_x  : float
            e_norm_y  : float
            e_norm_z  : float
            e_top_x   : float
            e_top_y   : float
            e_top_z   : float
            e_height  : float  # um
            e_width   : float  # um
            e_thick   : float  # um
            """

    class Detector(dj.Part):
        definition = """  # subtable of Geometry
            -> master
            detector   : int
            ----
            -> DField
            d_center_x : float  # um
            d_center_y : float  # um
            d_center_z : float  # um
            d_norm_x   : float
            d_norm_y   : float
            d_norm_z   : float
            d_top_x    : float
            d_top_y    : float
            d_top_z    : float
            d_height   : float  # um
            d_width    : float  # um
            d_thick    : float  # um
            """

    def make(self, key):
        self.insert1(key)

        efields, dfields = (Design & key).fetch1("efields", "dfields")
        gm_file = (Design & key).fetch1("geometry_file")
        #df = pd.read_csv("./workdir/" + gm_file)
        df = pd.read_csv(gm_file)

        df_emitter = df[df["BoxType"] == "Emitter"]
        df_detector = df[df["BoxType"] == "Detector"]

        detector_entry = []
        for i, data in df_detector.iterrows():
            d_center_x, d_center_y, d_center_z = [
                float(x) for x in data["center"][1:-1].split(",")
            ]
            d_norm_x, d_norm_y, d_norm_z = [float(x) for x in data["normal"][1:-1].split(",")]
            detector_entry.append(
                dict(
                    key,
                    detector=i,
                    dsim=dfields,
                    d_center_x=d_center_x,
                    d_center_y=d_center_y,
                    d_center_z=d_center_z,
                    d_norm_x=d_norm_x,
                    d_norm_y=d_norm_y,
                    d_norm_z=d_norm_z,
                    d_top_x=0,
                    d_top_y=0,
                    d_top_z=1,
                    d_height=data["h"],
                    d_width=data["w"],
                    d_thick=data["t"],
                )
            )
        self.Detector.insert(detector_entry)

        emitter_entry = []
        for i, data in df_emitter.iterrows():
            e_center_x, e_center_y, e_center_z = [
                float(x) for x in data["center"][1:-1].split(",")
            ]
            e_norm_x, e_norm_y, e_norm_z = [float(x) for x in data["normal"][1:-1].split(",")]

            emitter_entry.append(
                dict(
                    key,
                    emitter=i,
                    esim=efields,
                    e_center_x=e_center_x,
                    e_center_y=e_center_y,
                    e_center_z=e_center_z,
                    e_norm_x=e_norm_x,
                    e_norm_y=e_norm_y,
                    e_norm_z=e_norm_z,
                    e_top_x=0,
                    e_top_y=0,
                    e_top_z=1,
                    e_height=data["h"],
                    e_width=data["w"],
                    e_thick=data["t"],
                )
            )
        self.Emitter.insert(emitter_entry)
