import numpy as np
import pathlib
import itertools
import re
import tqdm
import datajoint as dj

schema = dj.schema('photonics')
schema.spawn_missing_classes()

@schema
class Design(dj.Lookup):
    definition = """
    design : smallint     # design number
    ---
    design_title : varchar(255)    
    design_description : varchar(1000)
    design_path    : varchar(255)  
    geometry_file  : varchar(255)
    center_offset  : blob   # offset from legacy implementation
    efields : blob   # efield selection
    dfields : blob   # dfield selection
    """
    
    contents = [
        (1, '8-emitter-design', '', 'Design1/matrix_8pix_random_1200x1200x400_15-04-17', 
         'geometry.csv', (600, 600, 0), 
         (0,), (0,)),
        (3, "Wesley Sacher's shaped fields", "", 
         "Design3/matrix_wesley1_revised_revised_1000x1000x1000_15-12-15", 
         "geometry_beams_as_emitters_wesley1_1000x1000x1000.csv", (550, 510, 0), 
         (10, 11, 12, 13, 14, 15, 16, 17, 18), (1,)),
        (4, "Shaped fields with 30-degree-collection cones", 
         "50 emitters per shank, 30-degree emission detection fields", 
         "Design4/matrix_steer_and_collect_a1_b3_v3_16-06-02", 
         "steer_coll_a1_b3_beams_as_emitters_geometry.csv", (550, 510, 0), 
         (10, 11, 12, 13, 14, 15, 16, 17, 18), (2,)),
        (5, "Wesley Sacher's shaped fields", "", 
         "Design3/matrix_wesley1_revised_revised_1000x1000x1000_15-12-15", 
         "geometry_beams_as_emitters_wesley1_1000x1000x1000.csv", (550, 510, 0), 
         (20, 21, 22, 23, 24, 25, 26, 27, 28), (1,)),
        (6, "Shaped fields with 30-degree-collection cones", 
         "50 emitters per shank, 30-degree emission detection fields", 
         "Design4/matrix_steer_and_collect_a1_b3_v3_16-06-02", 
         "steer_coll_a1_b3_beams_as_emitters_geometry.csv", (550, 510, 0), 
         (20, 21, 22, 23, 24, 25, 26, 27, 28), (3,)),
        (7, "Wesley Sacher's shaped fields", "", 
         "Design3/matrix_wesley1_revised_revised_1000x1000x1000_15-12-15", 
         "geometry_beams_as_emitters_wesley1_1000x1000x1000.csv", (550, 510, 0), 
         (30, 31, 32, 33, 34, 35, 36, 37, 38), (1,)),
        (8, "Shaped fields with 30-degree-collection cones", 
         "50 emitters per shank, 30-degree emission detection fields", 
         "Design4/matrix_steer_and_collect_a1_b3_v3_16-06-02", 
         "steer_coll_a1_b3_beams_as_emitters_geometry.csv", (550, 510, 0), 
         (30, 31, 32, 33, 34, 35, 36, 37, 38), (3,)),
        (9, "Shaped fields with 30-degree-collection cones",
         "50 emitters per shank, 30-degree emission detection fields",
         "Design4/matrix_steer_and_collect_a1_b3_v3_16-06-02",
         "steer_coll_a1_b3_beams_as_emitters_geometry.csv", (550, 510, 0),
         (10, 11, 12, 13, 14, 15, 16, 17, 18), (2,)),
        (10, "Shaped fields with 30-degree-collection cones",
         "50 emitters per shank, 30-degree emission detection fields",
         "Design4/matrix_steer_and_collect_a1_b3_v3_16-06-02",
         "steer_coll_a1_b3_beams_as_emitters_geometry.csv", (550, 510, 0),
         (10, 11, 12, 13, 14, 15, 16, 17, 18), (4,)),
    ]


@schema
class Geometry(dj.Imported):
    definition = """
    -> Design
    ---
    """
    
    class Emitter(dj.Part):
        definition = """  # subtable of Geometry
            -> master
            emitter :smallint
            ----
            -> EField
            e_center_x   :float  # um
            e_center_y   :float  # um
            e_center_z   :float  # um
            e_norm_x : float 
            e_norm_y : float 
            e_norm_z : float 
            e_top_x : float 
            e_top_y : float 
            e_top_z : float 
            e_height : float  # um
            e_width  : float  # um
            e_thick  : float  # um            
            """
        
    class Detector(dj.Part):
        definition = """    # subtable of Geometry
            -> master
            detector :smallint
            ----
            -> DField
            d_center_x   :float  # um
            d_center_y   :float  # um
            d_center_z   :float  # um
            d_norm_x : float 
            d_norm_y : float 
            d_norm_z : float 
            d_top_x : float 
            d_top_y : float 
            d_top_z : float 
            d_height : float  # um
            d_width  : float  # um
            d_thick  : float  # um            
            """

    def make(self, key):
        self.insert1(key)
        legacy_filepath = '../legacy/matrices'
        detector_pattern = re.compile(r'Detector,"\((?P<center>.*)\)","\((?P<normal>.*)\)","\((?P<top>.*)\)",(?P<height>.*),(?P<width>.*),(?P<thick>.*)')
        emitter_pattern = re.compile(r'Emitter,"\((?P<center>.*)\)","\((?P<normal>.*)\)","\((?P<top>.*)\)",(?P<height>.*),(?P<width>.*),(?P<thick>.*)')
        d_count = itertools.count()
        e_count = itertools.count()
        efields, dfields = (Design & key).fetch1('efields', 'dfields')
        last_rec = {}
        origin = (Design & key).fetch1('center_offset')
        for line in pathlib.Path(legacy_filepath, *(Design & key).fetch1('design_path', 'geometry_file')).open():
            # detectors
            match = detector_pattern.match(line)
            if match:
                rec = dict(key, detector=next(d_count))
                rec.update(zip(('d_center_x','d_center_y','d_center_z'), 
                               (float(i)-offset for i, offset in zip(match['center'].split(','), origin))))
                if key['design'] == 1:
                    rec.update(zip(('d_norm_x', 'd_norm_y', 'd_norm_z'),
                                   (float(i) for i in match['normal'].split(','))))
                    rec.update(zip(('d_top_x', 'd_top_y', 'd_top_z'),
                                   (float(i) for i in match['top'].split(','))))
                else:
                    azimuth = (rec['d_center_z'] - 5)*np.pi*9/40 + np.pi/16
                    rec.update(d_norm_x=np.cos(azimuth), 
                               d_norm_y=np.sin(azimuth), 
                               d_norm_z=0,
                               d_top_x=0,
                               d_top_y=0,
                               d_top_z=1)
                rec.update(
                    d_height=float(match['height']), 
                    d_width=float(match['width']),
                    d_thick=float(match['thick']))
                if rec != last_rec:
                    self.Detector().insert(dict(rec, dsim=dfield, detector=next(d_count))
                                           for dfield in dfields)
                    last_rec = rec
                continue
                
            # emitters
            match = emitter_pattern.match(line)
            if match:
                rec = dict(key)
                rec.update(zip(('e_center_x', 'e_center_y', 'e_center_z'),
                               (float(i)-offset for i, offset in zip(match['center'].split(','), origin))))
                if key['design'] == 1:
                    rec.update(zip(('e_norm_x', 'e_norm_y', 'e_norm_z'),
                                   (float(i) for i in match['normal'].split(','))))
                    rec.update(zip(('e_top_x', 'e_top_y', 'e_top_z'),
                                   (float(i) for i in match['top'].split(','))))
                else:
                    azimuth = (rec['e_center_z'] - 5)*np.pi*9/40 + np.pi/16
                    if key['design'] >= 9:
                        azimuth += np.pi
                    rec.update(e_norm_x=np.cos(azimuth),
                               e_norm_y=np.sin(azimuth), 
                               e_norm_z=0,
                               e_top_x=0,
                               e_top_y=0,
                               e_top_z=1)

                rec.update(
                    e_height=float(match['height']), 
                    e_width=float(match['width']),
                    e_thick=float(match['thick']))
                if rec != last_rec:
                    self.Emitter().insert(
                        dict(rec, esim=efield, emitter=next(e_count))
                        for efield in efields)
                    last_rec = rec
                continue
