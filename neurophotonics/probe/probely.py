import time
import functools
import numpy as np
import pandas as pd
from scipy.linalg import norm
from scipy.spatial.transform import Rotation as R
import plotly.graph_objects as go


def timer(func):
    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        tic = time.perf_counter()
        value = func(*args, **kwargs)
        toc = time.perf_counter()
        elapsed_time = toc - tic
        print(f"Completed {func.__name__} in {elapsed_time:0.4f} sec.")
        return value

    return wrapper_timer


class Square:
    def __init__(self, h, w, name="", is_emitter=True):
        self.height = h
        self.width = w
        self.name = name
        self.is_emitter = is_emitter

        if np.all([h, w]):
            self.make_vertices()

    def make_vertices(self):
        self.tl = np.array([-self.width / 2.0, 0.0, self.height / 2.0])  # top left
        self.bl = np.array([-self.width / 2.0, 0.0, -self.height / 2.0])  # bottom left
        self.br = np.array([self.width / 2.0, 0.0, -self.height / 2.0])  # bottom right
        self.tr = np.array([self.width / 2.0, 0.0, self.height / 2.0])  # top right

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

    def rotate_normal(self, seq="x", angles=45):
        r = R.from_euler(seq, angles, degrees=True)
        self.n = r.apply(self.n)

    def translate(self, r):
        self.tl += r
        self.bl += r
        self.br += r
        self.tr += r
        self.centroid += r

    def rotate_around(self, vertices, angle=45, seq="z"):
        mean_r = (self.__getattribute__(vertices[0]) + self.__getattribute__(vertices[1])) / 2
        self.translate(-mean_r)
        self.rotate(seq, angle)
        self.translate(mean_r)

    def vertices_positions(self, precision=13):
        xs = np.round([self.tl[0], self.bl[0], self.br[0], self.tr[0]], precision)
        ys = np.round([self.tl[1], self.bl[1], self.br[1], self.tr[1]], precision)
        zs = np.round([self.tl[2], self.bl[2], self.br[2], self.tr[2]], precision)
        return xs, ys, zs

    def plot_squares2d(
        self,
        fill="toself",
        fill_color="#64fbbd",
        mode="lines",
        line=dict(color="gray"),
        fig=None,
        rotate=False,
    ):
        if not fig:
            fig = go.Figure()

        xs, _, zs = self.vertices_positions()
        if rotate:
            xs, zs = zs, xs

        fig.add_traces(
            go.Scatter(
                x=xs,
                y=zs,
                fill=fill,
                fillcolor=fill_color,
                mode=mode,
                line=line,
            )
        )

    def plot_squares3d(self, fig=None, show=False, surfacecolor="turquoise", cone=False):
        if not fig:
            fig = go.Figure()

        xs, ys, zs = self.vertices_positions()

        xs = np.append(xs, xs[0])
        ys = np.append(ys, ys[0])
        zs = np.append(zs, zs[0])

        fig.add_traces(
            go.Scatter3d(
                x=xs,
                y=ys,
                z=zs,
                line=dict(color="#000000"),
                mode="lines",
                name="",
                surfaceaxis=1,
                surfacecolor=surfacecolor,
                showlegend=False,
            )
        )

        if cone:
            size_ref = np.mean([norm(self.bl - self.br), norm(self.br - self.tr)]) * 0.05
            fig.add_traces(
                go.Cone(
                    x=[
                        self.centroid[0] + self.n[0] * size_ref,
                    ],
                    y=[
                        self.centroid[1] + self.n[1] * size_ref,
                    ],
                    z=[
                        self.centroid[2] + self.n[2] * size_ref,
                    ],
                    u=[
                        self.n[0],
                    ],
                    v=[
                        self.n[1],
                    ],
                    w=[
                        self.n[2],
                    ],
                    sizemode="absolute",
                    sizeref=size_ref,
                    anchor="tip",
                )
            )

        if show:
            fig.show()


class Probe:
    def __init__(
        self,
        probe_dimensions=[1200, 120, 1300],
        n_e_box=[5, 60],
        e_box_length=10,
        e_box_sep=10,
        e_box_vertical_margin=15,
        e_box_horizontal_margin=5,
        n_d_box=[22, 240],
        d_box_length=5,
        d_box_sep=0,
        d_box_vertical_margin=5,
        d_box_horizontal_margin=0,
        name="",
    ):
        self.name = name

        self.height = probe_dimensions[0]  # height of probe
        self.width = probe_dimensions[1]  # width of probe
        self.tip = probe_dimensions[2]  # length between flat bottom and tip
        self.make_vertices()

        if n_e_box[0]:
            n_e_box[0] = int(
                np.floor(
                    (self.width - 2 * e_box_horizontal_margin + e_box_sep)
                    / (e_box_length + e_box_sep)
                )
            )
        if n_e_box[1]:
            n_e_box[1] = int(
                np.floor(
                    (self.height - 2 * e_box_vertical_margin + e_box_sep)
                    / (e_box_length + e_box_sep)
                )
            )

        if n_d_box[0]:
            n_d_box[0] = int(
                np.floor(
                    (self.width - 2 * d_box_horizontal_margin + d_box_sep)
                    / (d_box_length + d_box_sep)
                )
            )
        if n_d_box[1]:
            n_d_box[1] = int(
                np.floor(
                    (self.height - 2 * d_box_vertical_margin + d_box_sep)
                    / (d_box_length + d_box_sep)
                )
            )

        if n_e_box:
            self.n_e_box = n_e_box
            self.e_box_length = e_box_length
            self.e_box_sep = e_box_sep
            self.e_box_vertical_margin = e_box_vertical_margin
            self.e_box_horizontal_margin = e_box_horizontal_margin
            self.init_e_boxes()

        if n_e_box:
            self.n_d_box = n_d_box
            self.d_box_length = d_box_length
            self.d_box_sep = d_box_sep
            self.d_box_vertical_margin = d_box_vertical_margin
            self.d_box_horizontal_margin = d_box_horizontal_margin
            self.init_d_boxes()

    def update_probe_dimensions(
        self,
        probe_dimensions,
    ):
        if len(probe_dimensions) == 3 and np.sum(self.centroid) == 0.0:
            self.height = probe_dimensions[0]
            self.width = probe_dimensions[1]
            self.tip = probe_dimensions[2]
            self.make_vertices()

    def make_vertices(self):
        self.tl = np.array([-self.width / 2.0, 0.0, self.height / 2.0])  # top left
        self.bl = np.array([-self.width / 2.0, 0.0, -self.height / 2.0])  # bottom left
        self.br = np.array([self.width / 2.0, 0.0, -self.height / 2.0])  # bottom right
        self.tr = np.array([self.width / 2.0, 0.0, self.height / 2.0])  # top right
        self.tip = np.array([0.0, 0.0, -self.tip / 2.0])  # tip

        self.centroid = np.array([0.0, 0.0, 0.0])
        self.n = np.array([0.0, 1.0, 0.0])  # normal vector of the surface

    def rotate(self, seq="x", angles=90):
        r = R.from_euler(seq, angles, degrees=True)
        self.tl = r.apply(self.tl)
        self.bl = r.apply(self.bl)
        self.br = r.apply(self.br)
        self.tr = r.apply(self.tr)
        self.tip = r.apply(self.tip)
        self.n = r.apply(self.n)
        self.centroid = r.apply(self.centroid)

        if self.e_pixels:
            [e_pixel.rotate(seq, angles) for e_pixel in self.e_pixels]
        if self.d_pixels:
            [d_pixel.rotate(seq, angles) for d_pixel in self.d_pixels]

    def translate(self, r):
        self.tl += r
        self.bl += r
        self.br += r
        self.tr += r
        self.tip += r
        self.centroid += r

        if self.e_pixels:
            [e_pixel.translate(r) for e_pixel in self.e_pixels]
        if self.d_pixels:
            [d_pixel.translate(r) for d_pixel in self.d_pixels]

    def rotate_around(self, vertices, angle=45, seq="z"):
        mean_r = (self.__getattribute__(vertices[0]) + self.__getattribute__(vertices[1])) / 2

        self.translate(-mean_r)
        self.rotate(seq, angle)
        self.translate(mean_r)

    def vertices_positions(self, precision=13):
        xs = np.round([self.br[0], self.tr[0], self.tl[0], self.bl[0], self.tip[0]], precision)
        ys = np.round([self.br[1], self.tr[1], self.tl[1], self.bl[1], self.tip[1]], precision)
        zs = np.round([self.br[2], self.tr[2], self.tl[2], self.bl[2], self.tip[2]], precision)
        return xs, ys, zs

    def e_centroids(self):
        # Initial positions of the E-pixels
        xs = np.array(
            [
                self.e_box_horizontal_margin
                + self.e_box_length / 2
                + i * (self.e_box_length + self.e_box_sep)
                for i in range(self.n_e_box[0])
            ]
        )
        xs -= xs.mean()

        zs = np.array(
            [
                self.e_box_vertical_margin
                + self.e_box_length / 2
                + i * (self.e_box_length + self.e_box_sep)
                for i in range(self.n_e_box[1])
            ]
        )
        zs -= zs.mean()
        return [[x, 0, z] for x in xs for z in zs]

    def d_centroids(self):
        xs = np.array(
            [
                self.d_box_horizontal_margin
                + self.d_box_length / 2
                + i * (self.d_box_length + self.d_box_sep)
                for i in range(self.n_d_box[0])
            ]
        )
        xs -= xs.mean()
        zs = np.array(
            [
                self.d_box_vertical_margin
                + self.d_box_length / 2
                + i * (self.d_box_length + self.d_box_sep)
                for i in range(self.n_d_box[1])
            ]
        )
        zs -= zs.mean()

        candidates = [[x, 0, z] for x in xs for z in zs]
        e_pixels = self.e_centroids()
        e_xs = set([centroid[0] for centroid in e_pixels])
        e_zs = set([centroid[2] for centroid in e_pixels])
        x_overlaps = [x for x in xs for ex in e_xs if (abs(x - ex) <= self.e_box_length / 2)]
        z_overlaps = [z for z in zs for ez in e_zs if (abs(z - ez) <= self.e_box_length / 2)]
        return [
            candidate
            for candidate in candidates
            if not ((candidate[0] in x_overlaps) and (candidate[2] in z_overlaps))
        ]

    def init_e_boxes(self):
        coords = self.e_centroids()
        self.e_pixels = [Square(self.e_box_length, self.e_box_length, True) for coor in coords]
        [i[0].translate(i[1]) for i in zip(self.e_pixels, coords)]

    def init_d_boxes(self):
        coords = self.d_centroids()
        self.d_pixels = [Square(self.d_box_length, self.d_box_length, False) for coor in coords]
        [i[0].translate(i[1]) for i in zip(self.d_pixels, coords)]

    def plot_2d(self, e_pixels=True, d_pixels=True, fig=None, show=False, rotate=True):
        if not fig:
            fig = go.Figure()

        # Shank
        xs, _, zs = self.vertices_positions(precision=13)
        if rotate:
            xs, zs = zs, xs
        xs = np.append(xs, xs[0])
        zs = np.append(zs, zs[0])

        fig.add_traces(
            go.Scatter(
                x=xs,
                y=zs,
                fill="tonexty",
                mode="lines",
                line=dict(color="black"),
                fillcolor="#bfbfbf",
            )
        )

        # E-pixels
        if e_pixels:
            [
                e_p.plot_squares2d(
                    fig=fig,
                    fill_color="#00c1df",
                    mode="lines",
                    line=dict(color="#759196"),
                    rotate=rotate,
                )
                for e_p in self.e_pixels
            ]

        # D-pixels
        if d_pixels:
            [
                d_p.plot_squares2d(
                    fig=fig,
                    fill_color="#64fbbd",
                    mode="lines",
                    line=dict(color="#759196"),
                    rotate=rotate,
                )
                for d_p in self.d_pixels
            ]

        fig.update_layout(showlegend=False)
        if show:
            fig.show()

    def plot_probe3d(
        self,
        fig=None,
        show=False,
        probe_surfacecolor="gray",
        e_pixel_surfacecolor="turquoise",
        d_pixel_surfacecolor="blue",
        cone=True,
        e_pixels=False,
        d_pixels=False,
    ):
        if not fig:
            fig = go.Figure()

        if cone:
            size_ref = np.mean([norm(self.bl - self.br), norm(self.br - self.tr)]) * 0.05
            fig.add_traces(
                go.Cone(
                    x=[
                        self.centroid[0] + self.n[0] * size_ref,
                    ],
                    y=[
                        self.centroid[1] + self.n[1] * size_ref,
                    ],
                    z=[
                        self.centroid[2] + self.n[2] * size_ref,
                    ],
                    u=[
                        self.n[0],
                    ],
                    v=[
                        self.n[1],
                    ],
                    w=[
                        self.n[2],
                    ],
                    sizemode="absolute",
                    sizeref=size_ref,
                    anchor="tip",
                )
            )

        xs, ys, zs = self.vertices_positions()
        xs = np.append(xs, xs[0])
        ys = np.append(ys, ys[0])
        zs = np.append(zs, zs[0])

        # for x, y, z in zip(xs, ys, zs):
        # Shank
        fig.add_traces(
            go.Scatter3d(
                x=xs,
                y=ys,
                z=zs,
                line=dict(color="#000000"),
                mode="lines",
                name="",
                surfaceaxis=1,
                surfacecolor=probe_surfacecolor,
                showlegend=False,
            )
        )

        # E-pixels
        if e_pixels:
            [
                e_pixel.plot_squares3d(fig=fig, surfacecolor=e_pixel_surfacecolor)
                for e_pixel in self.e_pixels
            ]

        # D-pixels
        if d_pixels:
            [
                d_pixel.plot_squares3d(fig=fig, surfacecolor=d_pixel_surfacecolor)
                for d_pixel in self.d_pixels
            ]

        delta = self.height / 2 + 200
        fig.update_layout(
            scene=dict(
                xaxis=dict(range=[self.centroid[0] - delta, self.centroid[0] + delta]),
                yaxis=dict(range=[self.centroid[1] - delta, self.centroid[1] + delta]),
                zaxis=dict(range=[self.centroid[2] - delta, self.centroid[2] + delta]),
            ),
        )
        fig.update_layout(coloraxis_showscale=False)

        config = {"displayModeBar": True}
        if show:
            fig.show(config=config)


class ProbeGroup:
    def __init__(self, probes):
        self.n_probes = len(probes)
        self.probes = probes

    def add_probe(self, probe):
        self.probes.append(probe)

    def plot_shanks(self, cone=True, show=False, e_pixels=False, d_pixels=False, fig=None):
        if not fig:
            fig = go.Figure()

        for probe in self.probes:
            probe.plot_probe3d(fig=fig, cone=cone, e_pixels=e_pixels, d_pixels=d_pixels)

        if self.n_probes == 1:
            probe0 = self.probes[0]
            delta = probe0.height / 2 + 200
            fig.update_layout(
                scene=dict(
                    xaxis=dict(range=[probe0.centroid[0] - delta, probe0.centroid[0] + delta]),
                    yaxis=dict(range=[probe0.centroid[1] - delta, probe0.centroid[1] + delta]),
                    zaxis=dict(range=[probe0.centroid[2] - delta, probe0.centroid[2] + delta]),
                ),
            )
        elif not np.diff([probe.centroid for probe in self.probes], 1, 0).sum():
            probe0 = self.probes[0]
            delta = probe0.height / 2 + 200
            fig.update_layout(
                scene=dict(
                    xaxis=dict(range=[probe0.centroid[0] - delta, probe0.centroid[0] + delta]),
                    yaxis=dict(range=[probe0.centroid[1] - delta, probe0.centroid[1] + delta]),
                    zaxis=dict(range=[probe0.centroid[2] - delta, probe0.centroid[2] + delta]),
                )
            )
        else:
            big_list = [probe.width for probe in self.probes] + [
                probe.height for probe in self.probes
            ]
            delta = np.max(big_list)
            centroid = np.asarray([probe.centroid for probe in self.probes]).mean(0)
            fig.update_layout(
                scene=dict(
                    xaxis=dict(range=[centroid[0] - delta, centroid[0] + delta]),
                    yaxis=dict(range=[centroid[1] - delta, centroid[1] + delta]),
                    zaxis=dict(range=[centroid[2] - delta, centroid[2] + delta]),
                )
            )
        if show:
            fig.show()

    def to_df(self):
        df = pd.DataFrame(columns=["BoxType", "center", "normal", "top", "h", "w", "t"])

        df["BoxType"] = ["Emitter" for probe in self.probes for e_pixel in probe.e_pixels] + [
            "Detector" for probe in self.probes for d_pixel in probe.d_pixels
        ]
        df["center"] = [
            tuple(e_pixel.centroid) for probe in self.probes for e_pixel in probe.e_pixels
        ] + [tuple(d_pixel.centroid) for probe in self.probes for d_pixel in probe.d_pixels]
        df["normal"] = [
            tuple(e_pixel.n) for probe in self.probes for e_pixel in probe.e_pixels
        ] + [tuple(d_pixel.n) for probe in self.probes for d_pixel in probe.d_pixels]
        df["h"] = [e_pixel.height for probe in self.probes for e_pixel in probe.e_pixels] + [
            d_pixel.height for probe in self.probes for d_pixel in probe.d_pixels
        ]
        df["w"] = [e_pixel.width for probe in self.probes for e_pixel in probe.e_pixels] + [
            d_pixel.width for probe in self.probes for d_pixel in probe.d_pixels
        ]
        df["t"] = 0

        return df
