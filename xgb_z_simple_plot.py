import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle
from cmcrameri import cm
import cartopy.crs as ccrs
import contextily as ctx
from pyproj import Transformer
from datetime import datetime


plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 14
# USER INPUTS
city = 'ghent'
date_time = '2021-07-31T11:00'  # Format: 'YYYY-MM-DDTHH:MM'
output_file = f'temp_plot_{city}_{date_time.replace(":", "").replace("-", "")}.png'
# Extract year from datetime
year = int(date_time.split('-')[0])
# Load data
ds = xr.open_dataset(f"grid_netcdf/{city}_{year}_combined.nc")
temp = ds.t2m.sel(time=date_time, method='nearest')
lcz_grid = ds.lcz.values 
x_vals = ds.x.values
y_vals = ds.y.values
ds.close()

# Setup projection and coordinates
proj_3035 = ccrs.LambertAzimuthalEqualArea(
    central_longitude=10.0, central_latitude=52.0,
    false_easting=4321000.0, false_northing=3210000.0
)
transformer = Transformer.from_crs("EPSG:3035", "EPSG:4326", always_xy=True)
xx, yy = np.meshgrid(x_vals, y_vals)
lons, lats = transformer.transform(xx.ravel(), yy.ravel())
lons = lons.reshape(xx.shape)
lats = lats.reshape(yy.shape)
# Mask water (LCZ 17)
temp_masked = np.ma.array(temp.values, mask=(lcz_grid == 17))
# Create plot
fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={'projection': proj_3035})
xmin, xmax = float(x_vals.min()), float(x_vals.max())
ymin, ymax = float(y_vals.min()), float(y_vals.max())
ax.set_extent([xmin, xmax, ymin, ymax], crs=proj_3035)
# Plot temperature
cs = ax.pcolormesh(lons, lats, temp_masked, cmap=cm.roma_r,
                   transform=ccrs.PlateCarree(), shading='auto', alpha=0.6)
# Add basemap and colorbar
ctx.add_basemap(ax, crs=proj_3035, source=ctx.providers.CartoDB.Positron)
ax.set_title(f'{city.capitalize()} - {date_time}')
cbar = plt.colorbar(cs, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label('Temperature [°C]')
# Add scale bar (5 km)
scale_length = 5000
padding_x = (xmax - xmin) * 0.02
padding_y = (ymax - ymin) * 0.02
bar_x = xmax - scale_length - padding_x
bar_y = ymin + padding_y
bar_height = (ymax - ymin) * 0.005
scale_bar = Rectangle((bar_x, bar_y), scale_length, bar_height,
                      facecolor='black', transform=proj_3035, zorder=6)
ax.add_patch(scale_bar)
ax.text(bar_x + scale_length/2, bar_y + bar_height + padding_y*0.2,
        '5 km', ha='center', va='bottom', fontsize=10,
        transform=proj_3035, zorder=6)
plt.savefig(output_file, dpi=300, bbox_inches='tight')
plt.close()
print(f"Saved: {output_file}")