from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation
from PIL import Image

image_path = "imagenes\\pikachu3.jpg"
im = Image.open(image_path)
r, g, b = list(im.getdata(0)), list(im.getdata(1)), list(im.getdata(2))


fig = plt.figure()

axis = fig.add_subplot(1, 1, 1, projection="3d") # 3D plot with scalar values in each axis
axis.scatter(r, g, b, c="black", marker="o", alpha = 0.01);

axis.set_xlabel("Red")
axis.set_ylabel("Green")
axis.set_zlabel("Blue")




def rotate(angle):
    axis.view_init(azim=angle)

print("Making animation")
rot_animation = animation.FuncAnimation(fig, rotate, frames=np.arange(0, 362, 2), interval=100)
rot_animation.save('rotation.gif', dpi=80, writer='pillow')


image_path = "imagenes\\pikachu3.jpg"















