import pickle
import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


class Shank:
    def __init__(self, h, w, t):
        self.h_i = h  # height
        self.w_i = w  # width
        self.t_i = t  # length between flat bottom and tip

        self.h = h  # height
        self.w = w  # width
        self.t = t  # length between flat bottom and tip
        if np.all([h, w, t]):
            self.make_vertices()

    def reset_position(self):
        self.h = self.h_i
        self.w = self.w_i
        self.t = self.t_i
        self.origin = np.array([0, 0, 0])
        self.make_vertices()

    def make_vertices(self):
        self.tl = np.array([-self.w/2, 0, self.h/2])  # top left
        self.bl = np.array([-self.w/2, 0, -self.h/2])  # bottom left
        self.br = np.array([self.w/2, 0, -self.h/2])  # bottom right
        self.tr = np.array([self.w/2, 0, self.h/2])  # top right
        self.tip = np.array([0, 0, self.t - self.h/2])  # tip

        self.origin = np.array([0, 0, 0])
        self.n = np.array([0, 1, 0])  # normal vector of the surface

    def rotate(self, seq='x', angles=90):
        r = R.from_euler(seq, angles, degrees=True)
        self.tl = r.apply(self.tl)
        self.bl = r.apply(self.bl)
        self.br = r.apply(self.br)
        self.tr = r.apply(self.tr)
        self.tip = r.apply(self.tip)
        self.n = r.apply(self.n)
        self.origin = r.apply(self.origin)

    def translate(self, r):
        self.tl += r
        self.bl += r
        self.br += r
        self.tr += r
        self.tip += r
        self.origin += r

    def vertice_positions(self, precision):
        xs = np.round(
            [self.tip[0], self.tl[0], self.bl[0], self.br[0], self.tr[0]], precision)
        ys = np.round(
            [self.tip[1], self.tl[1], self.bl[1], self.br[1], self.tr[1]], precision)
        zs = np.round(
            [self.tip[2], self.tl[2], self.bl[2], self.br[2], self.tr[2]], precision)
        return xs, ys, zs

    def plot_plane(self, precision=13):
        plt.figure('Shank', figsize=plt.figaspect(1)*1.5)
        ax = plt.subplot(111, projection='3d')

        xs, ys, zs = self.vertice_positions(precision=precision)
        ax.scatter(xs, ys, zs)

        # 1. create vertices from points
        verts = [list(zip(xs, ys, zs))]
        # 2. create 3d polygons and specify parameters
        srf = Poly3DCollection(verts, alpha=.25, facecolor='#800000')
        # 3. add polygon to the figure (current axes)
        plt.gca().add_collection3d(srf)
        ax.quiver3D(*self.origin, *(self.n), color=['r'], length=200)
        ax.set_adjustable("datalim")

        lim = - self.t_i/2
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_zlim(-lim, lim)
        plt.show()

    def add_boxes(self, n_box, w, h, pos):
        self.box = []


class ShankGroup:
    def __init__(self, n_shanks=3, h=2000, w=200, t=2400):
        self.n_shanks = n_shanks
        self.shank = [Shank(h, w, t) for i in range(self.n_shanks)]

    def plot_shanks(self):
        plt.figure('Shank', figsize=plt.figaspect(1)*1.5)
        ax = plt.subplot(111, projection='3d')

        for i in range(self.n_shanks):
            xs, ys, zs = self.shank[i].vertice_positions(precision=12)
            ax.scatter(xs, ys, zs)

            # 1. create vertices from points
            verts = [list(zip(xs, ys, zs))]
            # 2. create 3d polygons and specify parameters
            srf = Poly3DCollection(verts, alpha=.25, facecolor='#800000')
            # 3. add polygon to the figure (current axes)
            plt.gca().add_collection3d(srf)
            ax.quiver3D(*self.shank[i].origin, *(self.shank[i].n),
                        color=['r'], length=200)

        lim = self.shank[0].t_i/2
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_zlim(-lim, lim)
        plt.show()

    def save(self):
        file = open('filename_pi.obj', 'w')
