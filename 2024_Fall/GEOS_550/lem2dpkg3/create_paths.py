import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.path import Path
from matplotlib.widgets import PolygonSelector

dem = xr.open_dataarray("lds-nz-8m-DEM12-GTiff/dem_100x100.nc")

def onselect(verts):
    # Optionally, you can use Path to determine whether points lie inside the selection
    path = Path(verts)

fig,ax = plt.subplots(figsize=(7,5))
dem.plot(ax=ax,aspect=None,cmap="terrain")
# Set up the polygon selector
selector = PolygonSelector(ax, onselect=onselect, useblit=True)
plt.show()

# After figure is closed print the coordinates of the selected points
print("Selected points:")
vrt = np.array(selector.verts).T
x = np.around(vrt[0,:],2).tolist()
y = np.around(vrt[1,:],2).tolist()
print(x,y,sep="\n")