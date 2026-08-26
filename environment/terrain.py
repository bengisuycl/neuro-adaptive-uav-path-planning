# project_v1/environment/terrain.py

import rasterio
import numpy as np

class Terrain:
    def __init__(self, dem_path):
        self.dataset = rasterio.open(dem_path)
        self.data = self.dataset.read(1).astype(float)

        # Replace no-data values with NaN
        self.data[self.data == self.dataset.nodata] = np.nan

        # Precompute metadata
        self.transform = self.dataset.transform
        self.bounds = self.dataset.bounds
        self.crs = self.dataset.crs
        self.min_elevation = float(np.nanmin(self.data))
        self.max_elevation = float(np.nanmax(self.data))
        self.safe_altitude = self.max_elevation + 200.0

    def is_inside(self, x, y):
        """Check if (x, y) is within DEM bounds"""
        return (self.bounds.left <= x <= self.bounds.right and
                self.bounds.bottom <= y <= self.bounds.top)

    def get_height(self, x, y):
        """Get elevation at UTM (x, y) from DEM using row/col index"""
        try:
            row, col = self.dataset.index(x, y)
            if (0 <= row < self.data.shape[0]) and (0 <= col < self.data.shape[1]):
                return float(self.data[row, col])
            else:
                return np.nan
        except:
            return np.nan

    def shape(self):
        return self.data.shape
