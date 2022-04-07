from matplotlib import pyplot as plt
from matplotlib_scalebar.scalebar import ScaleBar
import numpy as np
import tqdm


class Space:
    def __init__(self, pitch=1, dims=(500, 500, 500),
                 emitter_shape="rect",
                 emitter_size=(10, 10, 0),
                 emitter_spread="lambertian",
                 anisotropy=0.88, absorption_length=14000, scatter_length=100,
                 beam_compression=1.0, beam_xy_aspect=1.0, y_steer=0.0,
                 detector_type='total'):

        # irradiation field grid
        self.pitch = pitch  # (um) voxel pitch
        self.dims = dims  # voxels
        self.emitter_shape = emitter_shape  # e.g. rect or sphere
        self.emitter_size = emitter_size  # (um) (width, height, depth) (x, y, z)
        self.detector_type = detector_type

        # simulated photon packets: positions and directions
        self.n = 2048  # photon packets at a time
        self.start_positions = np.empty((self.n, 3))  # (um) x, y, z
        self.positions = np.empty((self.n, 3))  # (um) x, y, z
        self.directions = np.empty((self.n, 3))  # (um) x, y, z

        # simulation parameters
        self.emitter_spread = emitter_spread
        self.mui = 1 / absorption_length  # absorption rate: 1/um
        self.mue = 1 / scatter_length  # scatter rate: 1/um
        self.mu = self.mui + self.mue  # total rate
        self.absorb_ratio = self.mui / self.mu  # probability of absoprtion at each hop
        self.anisotropy = anisotropy  # For the Henyey-Greenstein formula
        self.beam_compression = beam_compression
        self.beam_xy_aspect = beam_xy_aspect  # 2.0 means squeeze y by factor of 2.0
        self.y_steer = y_steer  # (radians) steer angle in the plane of the shank

        self.volume = np.zeros(self.dims, dtype='float32')  # x, y, z
        self.total_count = 0  # total emitted photons

    @property
    def emitter_area(self):
        return self.emitter_size[0] * self.emitter_size[1]

    def emit(self, photons_to_replace):
        """
        :param photons_to_replace: indices of the photons to generate
        """
        # generate positions
        m = photons_to_replace.sum()
        self.total_count += m
        if self.emitter_shape == "rect":
            positions = (np.random.rand(m, 3) - 0.5) * self.emitter_size  # initial position
        elif self.emitter_shape == "sphere":
            positions = np.random.randn(m, 3)
            positions *= 0.5 * np.array(self.emitter_size) / np.sqrt(
                (positions**2).sum(axis=1, keepdims=True))
        else:
            raise Exception('Invalid emitter shape')
        self.positions[photons_to_replace, :] = positions

        # generate directions
        if self.emitter_spread == "parallel":
            directions = 0, 0, 1  # beam along z axis
        elif self.emitter_spread == "spherical":
            directions = np.random.randn(m, 3)
            directions /= np.sqrt((directions**2).sum(axis=1, keepdims=True))
        elif self.emitter_spread == "lambertian":  # oriented along z-axis
            elev_sin = np.random.rand(m)**0.5   # sine of elevation
            azimuth = np.random.rand(m) * 2 * np.pi
            azi_cos, azi_sin = np.cos(azimuth), np.sin(azimuth)

            # beam compression
            if self.beam_compression < 1:
                elev_sin = np.sin(np.arcsin(elev_sin) * self.beam_compression)
            if self.beam_xy_aspect != 1:
                azi_sin = np.sin(np.arcsin(azi_sin) / self.beam_xy_aspect)
                azi_cos = np.sign(azi_cos) * np.sqrt(1 - azi_sin * azi_sin)
            vx, vy, vz = elev_sin * azi_cos, elev_sin * azi_sin, np.sqrt(1 - elev_sin * elev_sin)

            # beam steering around the x-axis
            if self.y_steer:
                steer_cos, steer_sin = np.cos(self.y_steer), np.sin(self.y_steer)
                vz, vy = steer_cos * vz - steer_sin * vy, steer_sin * vz + steer_cos * vy
            directions = np.stack((vx, vy, vz)).T
        else:
            raise Exception('invalid emission type')
        self.directions[photons_to_replace, :] = directions

        # retain photons' starting positions
        self.start_positions[photons_to_replace, :] = self.positions[photons_to_replace, :]

    def detector_sensitivity(self):
        """
        Detection selectivity based on incident angles self.directions and detector offsets self.detector_offsets
        """
        if self.detector_type == "total":  # use for emission field modeling
            return 1.0
        if self.detector_type == "one-sided":  # equivalent to a lambertian emitter
            return np.maximum(0, np.sign(self.directions[:, 2]))  # along z-axis
        if self.detector_type == "narrowed":
            return np.maximum(0, self.directions[:, 2])**4  # along z-axis
        if self.detector_type == "narrowed2":
            return np.maximum(0, self.directions[:, 2])**2  # along z-axis
        if self.detector_type == "narrowed8":
            return np.maximum(0, self.directions[:, 2])**8  # along z-axis
        raise Exception('Unknown detector type')

    def accumulate(self, start_points, end_points, lengths):
        """
        accumulate irradiation intensity between start_points and end_points
        """
        samples = 8  # points per hop
        for i in range(samples):
            fractions = np.random.rand(self.n)[:, None]
            dims = np.array(self.dims)
            positions = np.round(
                dims / 2 + (start_points * fractions + end_points * (1 - fractions)) / self.pitch
                ).astype(np.int32)
            keep = np.all(np.logical_and(positions >= 0, positions < dims), axis=1)
            positions = positions[keep, :]
            indices = np.ravel_multi_index((positions[:, 0], positions[:, 1], positions[:, 2]), self.volume.shape)
            self.volume.ravel()[indices] += (self.detector_sensitivity() * lengths)[keep] / samples / self.pitch**3

    def hop(self):
        """
        execute photon hops between events, updating the irradition field
        """
        # irradiate
        lengths = np.random.exponential(1. / self.mu, self.n)
        new_positions = self.positions + lengths[:, None] * self.directions
        self.accumulate(self.positions, new_positions, lengths)
        self.positions = new_positions

        # absorb or escape
        max_distance = 2000  # consider photons lost if they are more than 2 mm away
        absorbed = np.logical_or(
            np.random.rand(self.n) < self.absorb_ratio, (new_positions**2).sum(axis=1) > max_distance**2)
        self.emit(absorbed)  # eliminate and absorb new ones

        # scatter: gcos follows the Henyey-Greenstein distribution of cosines
        scattered = np.logical_not(absorbed)
        m = scattered.sum()
        g = self.anisotropy
        gcos = 0.5 / g * (1 + g * g - ((1 - g * g) / (1 - g + 2 * g * np.random.rand(m)))**2)
        gsin = np.sqrt(1 - gcos * gcos)
        v = self.directions[scattered] * np.sign(gcos[:, None])
        d = np.random.randn(m, 3) * 0.001
        d -= v * (v * d).sum(axis=1, keepdims=True)  # random orthogonal vector
        v += gsin[:, None] / np.sqrt((d**2).sum(axis=1, keepdims=True)) * d  # match length to gsin
        v /= np.sqrt((v**2).sum(axis=1, keepdims=True))  # normalize
        self.directions[scattered, :] = v

    def run(self, hops=100):
        self.emit(np.ones(self.n, dtype=np.bool))
        for _ in tqdm.tqdm(range(hops), miniters=20):
            self.hop()
        # at the end, the values in volumes depict the photon flux per um^2
        # per emitted photons
        self.volume /= self.total_count

    def plot(self, axis=None, title="", gamma=0.5, cmap='gray'):
        if axis is None:
            _, axis = plt.subplots(1, 1, figsize=(8, 8))
        axis.imshow((self.volume.sum(axis=0))**gamma, cmap=cmap)
        axis.axis(False)
        scale_bar = ScaleBar(self.pitch * 1e-6)
        axis.add_artist(scale_bar)
        title = f"{title}\n{self.total_count / 1e6:0.2f} million simulated photons"
        axis.set_title(title)
