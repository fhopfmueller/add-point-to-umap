import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.widgets import Slider
import umap

y = torch.load("./data/x_y_train.pt")[1]
x = torch.load("./data/x_y_train.pt")[0]

#x = x[0:2000, ...]
#y = y[0:2000, ...]

extra_point = .5*(x[0:1, ...] + x[1:2, ...])

u = umap.UMAP(verbose=2)
x_reshaped = x.view(-1, 28*28)
x_umapped = u.fit_transform(x_reshaped)

plt.ion()
fig = plt.figure()
ax_data = fig.add_subplot(131)
ax_data.scatter(x_umapped[:, 0], x_umapped[:, 1], c=y)
extra_point_u = u.transform(extra_point.view(-1, 28*28))
ax_data.scatter(extra_point_u[:, 0], extra_point_u[:, 1], s=200)
ax_img = fig.add_subplot(133)

ax_slider = fig.add_subplot(132)
slider = Slider(ax_slider, 'interpolate', 0, 1, valinit = .5)
def update(val):
    param = slider.val
    extra_point = param*x[0:1, ...] + (1-param)*x[1:2, ...]
    ax_img.imshow(extra_point[0,0,:,:])
    extra_point_u = u.transform(extra_point.view(-1, 28*28))
    ax_data.scatter(extra_point_u[:, 0], extra_point_u[:, 1], s=200)
    fig.canvas.draw()
slider.on_changed(update)
    
input("press enter...")

