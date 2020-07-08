import numpy as np
import datajoint as dj
from photonics.space import Space
from matplotlib import pyplot as plt

schema = dj.schema('photonics')


@schema
class DSim(dj.Lookup):
    definition = """
    # Detector Field Specification
    dsim : int
    --- 
    dsim_description  : varchar(1024)  
    detector_type='one-sided' : varchar(30)   # choice in simulation
    detector_width=10.00: decimal(5,2)   # (um) along x-axis
    detector_height=10.00:  decimal(5,2)  # (um) along y-axis
    anisotropy = 0.88 : float  # factor in the Henyey-Greenstein formula
    absorption_length = 14000 : float # (um)  average travel path before a absoprtion event 
    scatter_length=100 : float # (um) average travel path before a scatter event
    volume_dimx = 1000 : int unsigned # (voxels)
    volume_dimy = 1000 : int unsigned # (voxels)
    volume_dimz = 1000 : int unsigned # (voxels)
    pitch = 2.2 : float  # (um)  spatial sampling period of the model volume
    """

    contents = [
        dict(dsim=0, detector_type='one-sided', detector_height=50,
             dsim_description='100% Efficient Lambertian 10x50 rect'),
        dict(dsim=1, detector_type='one-sided', detector_height=20,
             dsim_description='100% Efficient Lambertian 10x20 rect'),
        dict(dsim=2, detector_type='narrowed', detector_height=20,
             dsim_description="Narrow selective as 4th power of cosine, 10x10 rect"),
        dict(dsim=3, detector_type='narrowed2', detector_height=20,
             dsim_description="Narrow selective as 2th power of cosine, 10x20 rect")]


@schema
class DField(dj.Computed):
    definition = """
    # Detector Field Reference Volume
    -> DSim
    ---
    volume : blob@photonics   # probability of a photon emitted at given point getting picked up by the given detector
    max_value : float   # should be < 1.0
    total_photons : int unsigned
    """

    def make(self, key):
        spec = (DSim & key).fetch1()

        kwargs = {k: spec[k] for k in spec if k in {
            'pitch', 'anisotropy', 'scatter_length', 'absorption_length', 'detector_type'}}

        kwargs.update(
            dims=tuple(spec[k] for k in ('volume_dimx', 'volume_dimy', 'volume_dimz')),
            emitter_spread='spherical',
            emitter_size=(float(spec['detector_width']), float(spec['detector_height']), 0))

        space = Space(**kwargs)
        space.run(hops=500_000)
        volume = space.volume * space.emitter_area
        self.insert1(dict(
            key,
            volume=np.float32(volume),
            max_value=volume.max(),
            total_photons=space.total_count))

    def plot(self, axis=None, gamma=0.7, cmap='gray_r', title=''):
        from matplotlib_scalebar.scalebar import ScaleBar
        info = (self * DSim).fetch1()
        if axis is None:
            _, axis = plt.subplots(1, 1, figsize=(8, 8))
        axis.imshow((info['volume'].sum(axis=0)) ** gamma, cmap=cmap)
        axis.axis(False)
        scale_bar = ScaleBar(info['pitch'] * 1e-6)
        axis.add_artist(scale_bar)
        title = f"{title}\n{info['total_photons'] / 1e6:0.2f} million simulated photons"
        axis.set_title(title)


@schema
class ESim(dj.Lookup):
    definition = """
    # Emission Field Specification
    esim : int
    --- 
    esim_description : varchar(1024) 
    beam_compression : float  
    y_steer : float   # the steer angle in the plane of the shank
    emitter_width=10.00: decimal(5,2)   # (um) along x-axis
    emitter_height=10.00:  decimal(5,2)  # (um) along y-axis
    anisotropy = 0.88 : float  # factor in the Henyey-Greenstein formula
    absorption_length = 14000 : float # (um)  average travel path before a absoprtion event    
    scatter_length=100 : float # (um) average travel path before a scatter event
    volume_dimx = 1000 : int unsigned # (voxels)
    volume_dimy = 1000 : int unsigned # (voxels)
    volume_dimz = 1000 : int unsigned # (voxels)
    beam_xy_aspect = 1.0 : float   # compression of y. E.g. 2.0 means that y is compressed by factor of 2
    pitch = 2.2 : float  # (um)  spatial sampling period of the model volume
    """

    contents = [
        dict(esim=0, esim_description="Lambertian 10 x 10",
             beam_compression=1.0, y_steer=0.0),

        dict(esim=10, esim_description="Narrowed to pi/4, steered -24/64*pi",
             beam_compression=1/4, y_steer=-24/64 * np.pi, beam_xy_aspect=1.0),
        dict(esim=11, esim_description="Narrowed to pi/4, steered -18/64*pi",
             beam_compression=1/4, y_steer=-18/64 * np.pi, beam_xy_aspect=1.0),
        dict(esim=12, esim_description="Narrowed to pi/4, steered -12/64*pi",
             beam_compression=1/4, y_steer=-12/64 * np.pi, beam_xy_aspect=1.0),
        dict(esim=13, esim_description="Narrowed to pi/4, steered -6/64*pi",
             beam_compression=1/4, y_steer=-6/64 * np.pi, beam_xy_aspect=1.0),
        dict(esim=14, esim_description="Narrowed to pi/4, steered 0",
             beam_compression=1/4, y_steer=0 * np.pi, beam_xy_aspect=1.0),
        dict(esim=15, esim_description="Narrowed to pi/4, steered +6/64*pi",
             beam_compression=1/4, y_steer=+6/64 * np.pi, beam_xy_aspect=1.0),
        dict(esim=16, esim_description="Narrowed to pi/4, steered +12/64*pi",
             beam_compression=1/4, y_steer=+12/64 * np.pi, beam_xy_aspect=1.0),
        dict(esim=17, esim_description="Narrowed to pi/4, steered +18/64*pi",
             beam_compression=1/4, y_steer=+18/64 * np.pi, beam_xy_aspect=1.0),
        dict(esim=18, esim_description="Narrowed to pi/4, steered +24/64*pi",
             beam_compression=1/3, y_steer=+24/64 * np.pi, beam_xy_aspect=2.0),

        dict(esim=20, esim_description="Narrowed to pi/3, steered -pi/3",
             beam_compression=1/3, y_steer=-np.pi / 3, beam_xy_aspect=1.0),
        dict(esim=21, esim_description="Narrowed to pi/3, steered -pi/4",
             beam_compression=1/3, y_steer=-np.pi / 4, beam_xy_aspect=1.0),
        dict(esim=22, esim_description="Narrowed to pi/3, steered -pi/6",
             beam_compression=1/3, y_steer=-np.pi / 6, beam_xy_aspect=1.0),
        dict(esim=23, esim_description="Narrowed to pi/3, steered -pi/12",
             beam_compression=1/3, y_steer=-np.pi / 12, beam_xy_aspect=1.0),
        dict(esim=24, esim_description="Narrowed to pi/3, steered 0",
             beam_compression=1/3, y_steer=0, beam_xy_aspect=1.0),
        dict(esim=25, esim_description="Narrowed to pi/3, steered +pi/12",
             beam_compression=1/3, y_steer=+np.pi / 12, beam_xy_aspect=1.0),
        dict(esim=26, esim_description="Narrowed to pi/3, steered +pi/6",
             beam_compression=1/3, y_steer=+np.pi / 6, beam_xy_aspect=1.0),
        dict(esim=27, esim_description="Narrowed to pi/3, steered +pi/4",
             beam_compression=1/3, y_steer=+np.pi / 4, beam_xy_aspect=1.0),
        dict(esim=28, esim_description="Narrowed to pi/3, steered +pi/3",
             beam_compression=1/3, y_steer=+np.pi / 3, beam_xy_aspect=1.0),

        dict(esim=30, esim_description="Narrowed to pi/3, steered -pi/3",
             beam_compression=1/3, y_steer=-np.pi / 3, beam_xy_aspect=2.0),
        dict(esim=31, esim_description="Narrowed to pi/3, steered -pi/4",
             beam_compression=1/3, y_steer=-np.pi / 4, beam_xy_aspect=2.0),
        dict(esim=32, esim_description="Narrowed to pi/3, steered -pi/6",
             beam_compression=1/3, y_steer=-np.pi / 6, beam_xy_aspect=2.0),
        dict(esim=33, esim_description="Narrowed to pi/3, steered -pi/12",
             beam_compression=1/3, y_steer=-np.pi / 12, beam_xy_aspect=2.0),
        dict(esim=34, esim_description="Narrowed to pi/3, steered 0",
             beam_compression=1/3, y_steer=0, beam_xy_aspect=2.0),
        dict(esim=35, esim_description="Narrowed to pi/3, steered +pi/12",
             beam_compression=1/3, y_steer=+np.pi / 12, beam_xy_aspect=2.0),
        dict(esim=36, esim_description="Narrowed to pi/3, steered +pi/6",
             beam_compression=1/3, y_steer=+np.pi / 6, beam_xy_aspect=2.0),
        dict(esim=37, esim_description="Narrowed to pi/3, steered +pi/4",
             beam_compression=1/3, y_steer=+np.pi / 4, beam_xy_aspect=2.0),
        dict(esim=38, esim_description="Narrowed to pi/3, steered +pi/3",
             beam_compression=1/3, y_steer=+np.pi / 3, beam_xy_aspect=2.0),
    ]


@schema
class EField(dj.Computed):
    definition = """
    # Emitter Field Reference Volume
    -> ESim
    ---
    volume : blob@photonics   # probability of a photon emitted at given point getting picked up by the given detector
    total_photons : int unsigned
    """

    def make(self, key):
        spec = (ESim & key).fetch1()

        # pass arguments from lookup to function
        kwargs = {k: spec[k] for k in spec if k in {
            'pitch', 'anisotropy', 'scatter_length',
            'y_steer', 'beam_compression', 'beam_xy_aspect',
            'absorption_length'}}

        kwargs.update(
            dims=tuple(spec[k] for k in ('volume_dimx', 'volume_dimy', 'volume_dimz')),
            emitter_size=(float(spec['emitter_width']), float(spec['emitter_height']), 0))

        space = Space(**kwargs)
        space.run(hops=500_000)
        self.insert1(dict(
            key,
            volume=np.float32(space.volume),
            total_photons=space.total_count))

    def plot(self, figsize=(8, 8), axis=None, gamma=0.7, cmap='magma', title=''):
        from matplotlib_scalebar.scalebar import ScaleBar
        info = (self * ESim).fetch1()
        if axis is None:
            _, axis = plt.subplots(1, 1, figsize=(8, 8))
        axis.imshow((info['volume'].sum(axis=0)) ** gamma, cmap=cmap)
        axis.axis(False)
        scale_bar = ScaleBar(info['pitch'] * 1e-6)
        axis.add_artist(scale_bar)
        title = f"{title}\n{info['total_photons'] / 1e6:0.2f} million simulated photons"
        axis.set_title(title)
