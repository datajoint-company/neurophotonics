import pickle
import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


class Shank:
    def __init__(self, shank_dimensions, box_dimensions=None, separation_dimensions=None):
        self.h = shank_dimensions[0]  # height of shank
        self.w = shank_dimensions[1]  # width of shank
        self.t = shank_dimensions[2]  # length between flat bottom and tip

        if np.all(shank_dimensions):
            self.make_vertices()

        if box_dimensions and separation_dimensions:
            self.bh = box_dimensions[0]  # height of box
            self.bw = box_dimensions[1]  # width of box
            self.sh = separation_dimensions[0] # vertical separation between boxes
            self.sw = separation_dimensions[1] # horizontal seraration between boxes

    def make_vertices(self):
        self.tl = np.array([-self.w/2, 0, self.h/2])  # top left
        self.bl = np.array([-self.w/2, 0, -self.h/2])  # bottom left
        self.br = np.array([self.w/2, 0, -self.h/2])  # bottom right
        self.tr = np.array([self.w/2, 0, self.h/2])  # top right
        self.tip = np.array([0, 0, self.t - self.h/2])  # tip

        self.centroid = np.array([0, 0, 0])
        self.n = np.array([0, 1, 0])  # normal vector of the surface

    def rotate(self, seq='x', angles=90):
        r = R.from_euler(seq, angles, degrees=True)
        self.tl = r.apply(self.tl)
        self.bl = r.apply(self.bl)
        self.br = r.apply(self.br)
        self.tr = r.apply(self.tr)
        self.tip = r.apply(self.tip)
        self.n = r.apply(self.n)
        self.centroid = r.apply(self.centroid)

    def translate(self, r):
        self.tl += r
        self.bl += r
        self.br += r
        self.tr += r
        self.tip += r
        self.centroid += r

    def vertice_positions(self, precision=13):
        xs = np.round(
            [self.br[0], self.tr[0], self.tl[0], self.bl[0], self.tip[0]], precision)
        ys = np.round(
            [self.br[1], self.tr[1], self.tl[1], self.bl[1], self.tip[1]], precision)
        zs = np.round(
            [self.br[2], self.tr[2], self.tl[2], self.bl[2], self.tip[2]], precision)
        return xs, ys, zs

    def plot_plane(self, precision=13, ax=None, show=False):
        if not ax:
            plt.figure('Shank', figsize=plt.figaspect(1)*1.5)
            ax = plt.subplot(111, projection='3d')

        xs, ys, zs = self.vertice_positions(precision=precision)
        ax.scatter(xs, ys, zs)

        # 1. create vertices from points
        verts = [list(zip(xs, ys, zs))]
        # 2. create 3d polygons and specify parameters
        srf = Poly3DCollection(verts, alpha=.5, facecolor='gray')
        srf.set_edgecolor('black')
        # 3. add polygon to the figure (current axes)
        plt.gca().add_collection3d(srf)
        ax.quiver3D(*self.centroid, *(self.n), color=['r'], length=200)
        ax.set_adjustable("datalim")

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        delta = self.h / 2
        ax.set_xlim(self.centroid[0] - delta, self.centroid[0] + delta)
        ax.set_ylim(self.centroid[1] - delta, self.centroid[1] + delta)
        ax.set_zlim(self.centroid[2] - delta, self.centroid[2] + delta)
        plt.show()

    def add_boxes(self, n_box, w, h, pos):
        self.box = []

    def make_grid(self):
        # find the greatest number of pixels that can fit on the width of the shank
        row_count = np.ceil((self.w + self.sw) / (self.bw + self.sw)) - 1
        # find the margin for each side of the row
        row_margin = (self.w - (row_count * self.bw + (row_count - 1) * self.sw)) / 2

        # find the greatest number of pixels that can fit on the length of the shank
        column_count = np.ceil((self.h + self.sh) / (self.bh + self.sh)) - 1
        # find the margin for each side of the column
        column_margin = (self.h - (column_count * self.bh + (column_count - 1) * self.sh)) / 2
        
        # calculate the x positions for the rows of boxes
        xs = np.arange(self.bl[0] + row_margin + self.bw / 2, self.w, self.bw + self.sw)
        # calculate the z positions for the columns of boxes
        zs = np.arange(self.bl[2] + column_margin + self.bh / 2, self.h, self.bh + self.sh)

        # zip xs and zs
        return [[x, 0, z] for x in xs for z in zs]



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
            ax.quiver3D(*self.shank[i].centroid, *(self.shank[i].n),
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
