import numpy as np
import scipy.ndimage
import neurophotonics as npx
from create_design import design1
import matplotlib.pyplot as plt
from matplotlib_scalebar.scalebar import ScaleBar


light_volume = npx.fields.EField.fetch1("volume").sum(0)  # This is centered at 500, 500, 500

PG = design1()

centroids0 = np.array([e_pixel.centroid for e_pixel in PG.probes[0].e_pixels])
centroids2 = np.array([e_pixel.centroid for e_pixel in PG.probes[2].e_pixels])

angle0 = np.arctan(PG.probes[0].e_pixels[0].n[1] / PG.probes[0].e_pixels[0].n[0]) * 180 / np.pi
angle2 = np.arctan(PG.probes[2].e_pixels[0].n[1] / PG.probes[2].e_pixels[0].n[0]) * 180 / np.pi

top_ten1 = centroids0[np.argsort(centroids0[:, -1])[:5]][:, :2]  # make it 2d
top_ten2 = centroids2[np.argsort(centroids2[:, -1])[:5]][:, :2]  # make it 2d


image = sum(
    [
        scipy.ndimage.shift(scipy.ndimage.rotate(light_volume, angle0, cval=1.6187507e-05), i)
        for i in top_ten2
    ]
    + [
        scipy.ndimage.shift(scipy.ndimage.rotate(light_volume, angle2, cval=1.6187507e-05), i)
        for i in top_ten1
    ]
)
image[image < 0] = 0

gamma = 0.7
plt.imshow(image[500:900, 600:1000] ** gamma, cmap="magma")
scalebar = ScaleBar(1e-6)
plt.gca().add_artist(scalebar)
plt.axis("off")
plt.show()

