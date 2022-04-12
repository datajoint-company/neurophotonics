import pickle
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


class Square:
    def __init__(self, h, w, name=""):
        self.h = h
        self.w = w
        self.name = name

        if np.all([h, w]):
            self.make_vertices()

    def make_vertices(self):
        self.tl = np.array([-self.w / 2, 0, self.h / 2])  # top left
        self.bl = np.array([-self.w / 2, 0, -self.h / 2])  # bottom left
        self.br = np.array([self.w / 2, 0, -self.h / 2])  # bottom right
        self.tr = np.array([self.w / 2, 0, self.h / 2])  # top right

        self.centroid = np.array([0.0, 0.0, 0.0])
        self.n = np.array([0.0, 1.0, 0.0])  # normal vector of the surface

    def rotate(self, seq="x", angles=90):
        r = R.from_euler(seq, angles, degrees=True)
        self.tl = r.apply(self.tl)
        self.bl = r.apply(self.bl)
        self.br = r.apply(self.br)
        self.tr = r.apply(self.tr)
        self.n = r.apply(self.n)
        self.centroid = r.apply(self.centroid)

    def translate(self, r):
        self.tl += r
        self.bl += r
        self.br += r
        self.tr += r
        self.centroid += r

    def vertices_positions(self, precision=13):
        xs = np.round([self.tl[0], self.bl[0], self.br[0], self.tr[0]], precision)
        ys = np.round([self.tl[1], self.bl[1], self.br[1], self.tr[1]], precision)
        zs = np.round([self.tl[2], self.bl[2], self.br[2], self.tr[2]], precision)
        return xs, ys, zs

    def plot_plane(self, precision=13, ax=None, show=False):
        if not ax:
            plt.figure("Box", figsize=plt.figaspect(1) * 1.5)
            ax = plt.subplot(111, projection="3d")

        xs, ys, zs = self.vertices_positions(precision=precision)
        # ax.scatter(xs, ys, zs)

        # 1. create vertices from points
        verts = [list(zip(xs, ys, zs))]
        # 2. create 3d polygons and specify parameters
        srf = Poly3DCollection(verts, alpha=0.25, facecolor="turquoise")
        srf.set_edgecolor("black")
        # 3. add polygon to the figure (current axes)
        plt.gca().add_collection3d(srf)
        ax.quiver3D(*self.centroid, *(self.n), color=["r"], length=200)
        # ax.set_adjustable("datalim")

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        delta = self.h / 2
        ax.set_xlim(self.centroid[0] - delta, self.centroid[0] + delta)
        ax.set_ylim(self.centroid[1] - delta, self.centroid[1] + delta)
        ax.set_zlim(self.centroid[2] - delta, self.centroid[2] + delta)

        if show:
            plt.show()


class Shank:
    def __init__(
        self, shank_dimensions, box_dimensions=None, separation_dimensions=None
    ):

        if len(shank_dimensions) == 3:
            self.h = shank_dimensions[0]  # height of shank
            self.w = shank_dimensions[1]  # width of shank
            self.t = shank_dimensions[2]  # length between flat bottom and tip
            self.make_vertices()

        if np.all(box_dimensions) and len(separation_dimensions) == 2:
            self.bh = box_dimensions[0]  # height of box
            self.bw = box_dimensions[1]  # width of box
            self.sh = separation_dimensions[0]  # vertical separation between boxes
            self.sw = separation_dimensions[1]  # horizontal seraration between boxes
            self.init_boxes()

    def make_vertices(self):
        self.tl = np.array([-self.w / 2, 0, self.h / 2])  # top left
        self.bl = np.array([-self.w / 2, 0, -self.h / 2])  # bottom left
        self.br = np.array([self.w / 2, 0, -self.h / 2])  # bottom right
        self.tr = np.array([self.w / 2, 0, self.h / 2])  # top right
        self.tip = np.array([0, 0, -self.t / 2])  # tip

        self.centroid = np.array([0, 0, 0])
        self.n = np.array([0, 1, 0])  # normal vector of the surface

    def rotate(self, seq="x", angles=90):
        r = R.from_euler(seq, angles, degrees=True)
        self.tl = r.apply(self.tl)
        self.bl = r.apply(self.bl)
        self.br = r.apply(self.br)
        self.tr = r.apply(self.tr)
        self.tip = r.apply(self.tip)
        self.n = r.apply(self.n)
        self.centroid = r.apply(self.centroid)

        if self.boxes:
            [box.rotate(seq, angles) for box in self.boxes]

    def translate(self, r):
        self.tl += r
        self.bl += r
        self.br += r
        self.tr += r
        self.tip += r
        self.centroid += r

        if self.boxes:
            [box.translate(r) for box in self.boxes]

    def vertices_positions(self, precision=13):
        xs = np.round(
            [self.br[0], self.tr[0], self.tl[0], self.bl[0], self.tip[0]], precision
        )
        ys = np.round(
            [self.br[1], self.tr[1], self.tl[1], self.bl[1], self.tip[1]], precision
        )
        zs = np.round(
            [self.br[2], self.tr[2], self.tl[2], self.bl[2], self.tip[2]], precision
        )
        return xs, ys, zs

    def plot_plane(self, precision=13, ax=None, show=False):
        if not ax:
            plt.figure("Shank", figsize=plt.figaspect(1) * 1.5)
            ax = plt.subplot(111, projection="3d")

        xs, ys, zs = self.vertices_positions(precision=precision)
        # ax.scatter(xs, ys, zs)

        # 1. create vertices from points
        verts = [list(zip(xs, ys, zs))]
        # 2. create 3d polygons and specify parameters
        srf = Poly3DCollection(verts, alpha=0.5, facecolor="gray")
        srf.set_edgecolor("black")
        # 3. add polygon to the figure (current axes)
        plt.gca().add_collection3d(srf)
        ax.quiver3D(*self.centroid, *(self.n), color=["r"], length=200)
        ax.set_adjustable("datalim")

        if "boxes" in self.__dict__.keys():
            for box in self.boxes:
                xs, ys, zs = box.vertices_positions(precision=precision)
                # ax.scatter(xs, ys, zs)

                # 1. create vertices from points
                verts = [list(zip(xs, ys, zs))]
                # 2. create 3d polygons and specify parameters
                srf = Poly3DCollection(verts, alpha=0.5, facecolor="turquoise")
                # 3. add polygon to the figure (current axes)
                plt.gca().add_collection3d(srf)
                ax.quiver3D(*self.centroid, *(self.n), color=["r"], length=200)
                ax.set_adjustable("datalim")

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        delta = self.h / 2
        ax.set_xlim(self.centroid[0] - delta, self.centroid[0] + delta)
        ax.set_ylim(self.centroid[1] - delta, self.centroid[1] + delta)
        ax.set_zlim(self.centroid[2] - delta, self.centroid[2] + delta)

        if show:
            plt.show()

    def plot_2d(self, show=False):
        fig, ax = plt.subplots(ncols=1, nrows=1)

        xs, _, zs = self.vertices_positions(precision=13)
        verts = list(zip(xs, zs))
        poly = plt.Polygon(verts, ec="k", fc="gray")
        ax.add_patch(poly)

        ax.scatter(xs, zs, s=0)

        for box in self.boxes:
            xs, _, zs = box.vertices_positions(precision=13)
            verts = list(zip(xs, zs))
            poly = plt.Polygon(verts, ec="k", fc="green")
            ax.add_patch(poly)
            ax.scatter(xs, zs, s=0)

        if show:
            plt.show()

    def init_boxes(self):
        # find the greatest number of pixels that can fit on the width of the shank
        row_count = np.ceil((self.w + self.sw) / (self.bw + self.sw)) - 1

        # find the margin for each side of the row
        row_margin = (self.w - (row_count * self.bw + (row_count - 1) * self.sw)) / 2

        # find the greatest number of pixels that can fit on the length of the shank
        column_count = np.ceil((self.h + self.sh) / (self.bh + self.sh)) - 1
        # find the margin for each side of the column
        column_margin = (
            self.h - (column_count * self.bh + (column_count - 1) * self.sh)
        ) / 2

        print(row_count)
        print(column_count)
        print(int(row_count))
        print(int(column_count))
        self.boxes = [
            Square(self.bh, self.bw)
            for i in range(int(row_count))
            for j in range(int(column_count))
        ]

        # calculate the x & z positions of boxes
        xs = np.arange(
            self.bl[0] + row_margin + self.bw / 2,
            self.w/2,
            self.bw + self.sw,
        )
        zs = np.arange(
            self.bl[2] + column_margin + self.bh / 2,
            self.h/2,
            self.bh + self.sh,
        )

        print(len(xs))
        print(len(zs))

        coords = [[x, 0, z] for x in xs for z in zs]
        # print(coords[0])

        [i[0].translate(i[1]) for i in zip(self.boxes, coords)]


class ShankGroup:
    def __init__(
        self,
        n_shanks=3,
        shank_dimensions=[1200, 100, 1300],
        box_dimensions=None,
        separation_dimensions=None,
        gname="",
        snames=None,
    ):
        self.n_shanks = n_shanks
        self.shank = [
            Shank(shank_dimensions, box_dimensions, separation_dimensions)
            for i in range(self.n_shanks)
        ]
        self.gname = gname

        if isinstance(snames, list):
            snames = snames
        else:
            snames = [snames]

        if len(snames) == self.n_shanks:
            for sname in snames:
                self.shank.name = sname

    def plot_shanks(self, ax=None, show=True):
        if not ax:
            plt.figure("ShankGroup", figsize=plt.figaspect(1) * 1.5)
            ax = plt.subplot(111, projection="3d")

        for shank in self.shank:
            shank.plot_plane(ax=ax)

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_adjustable("datalim")

        if not np.diff(
            [self.shank[i].centroid for i in range(self.n_shanks)], 1, 0
        ).sum():
            sh0 = self.shank[0]
            delta = sh0.h / 2
            ax.set_xlim(sh0.centroid[0] - delta, sh0.centroid[0] + delta)
            ax.set_ylim(sh0.centroid[1] - delta, sh0.centroid[1] + delta)
            ax.set_zlim(sh0.centroid[2] - delta, sh0.centroid[2] + delta)
        else:
            big_list = [shank.w for shank in self.shank] + [
                shank.h for shank in self.shank
            ]
            delta = np.max(big_list)
            centroid = np.asarray([shank.centroid for shank in self.shank]).mean(0)
            ax.set_xlim(centroid[0] - delta, centroid[0] + delta)
            ax.set_ylim(centroid[1] - delta, centroid[1] + delta)
            ax.set_zlim(centroid[2] - delta, centroid[2] + delta)

        if show:
            plt.show()

    def to_df(self):
        df = pd.DataFrame(columns=["BoxType", "center", "normal", "top", "h", "w", "t"])

        df["BoxType"] = [box.name for shank in self.shank for box in shank.boxes]
        df["center"] = [
            ('"' + str(tuple(box.centroid)) + '"')
            for shank in self.shank
            for box in shank.boxes
        ]
        df["normal"] = [
            ('"' + str(tuple(box.n)) + '"')
            for shank in self.shank
            for box in shank.boxes
        ]
        df["h"] = [box.h for shank in self.shank for box in shank.boxes]
        df["w"] = [box.w for shank in self.shank for box in shank.boxes]
        df["t"] = 0

        return df
