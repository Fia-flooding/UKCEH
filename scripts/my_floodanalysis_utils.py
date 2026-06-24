from __future__ import annotations

import io
import shutil
from pathlib import Path

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import Image, display
from matplotlib.colors import LinearSegmentedColormap

# -- Outlines colour map for rainfall fields ------------------------------------------
def custom_rainfall_cmap() -> LinearSegmentedColormap:
    """Return the project-standard cyan-blue-green-yellow-red-magenta-black cmap."""
    stops = [
        (0.00, 'cyan'),
        (0.18, 'blue'),
        (0.35, 'green'),
        (0.52, 'yellow'),
        (0.70, 'red'),
        (0.85, 'magenta'),
        (1.00, 'black'),
    ]
    return LinearSegmentedColormap.from_list('custom_rainbow', stops)


# ── Check which event files are present ──────────────────────────────────────
def check_files(events: list[dict], data_dir: Path) -> pd.DataFrame:
    rows = []
    for ev in events:
        eid = ev['event_id']
        mask_path = data_dir / f'catchment_mask_{eid}.csv'
        flat_path  = data_dir / f'rainfall_field_flat_{eid}.csv'
        rows.append({
            'event_id':   eid,
            'mask_found': mask_path.exists(),
            'flat_found': flat_path.exists(),
            'ready':      mask_path.exists() and flat_path.exists(),
        })
    return pd.DataFrame(rows)


# -- load the catchment mask and flat rainfall field for one event ------------------------------------------
def load_event_data(
    event_id: str,
    data_dir: Path,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load the catchment mask and flat rainfall field for one event.

    Returns
    -------
    flat_df   : DataFrame with columns [time, pixel_000000, pixel_000001, ...]
    values    : ndarray, shape (T, n_northings, n_eastings) – rainfall mm/hour
    mask      : ndarray, shape (n_northings, n_eastings) – NaN outside catchment
    northings : 1-D array of unique northing coordinates (metres, OSGB36)
    eastings  : 1-D array of unique easting coordinates  (metres, OSGB36)
    """
    mask_path = data_dir / f'catchment_mask_{event_id}.csv' # Creates the path to the catchment mask CSV file for the given event ID and names this path mask_path
    flat_path  = data_dir / f'rainfall_field_flat_{event_id}.csv' # Creates the path to the flattened rainfall field CSV file for the given event ID and names this path flat_path

    mask_df = pd.read_csv(mask_path) # Reads the catchment mask CSV file into a pandas DataFrame called mask_df
    flat_df = pd.read_csv(flat_path, parse_dates=['time']) # Reads the flattened rainfall field CSV file into a pandas DataFrame called flat_df, converting the 'time' column into datetime objects

    # Derive grid dimensions from the mask file
    northings = np.sort(mask_df['northing'].unique())[::-1] # Takes the Northing column --> removes duplicates = unique grid coords --> sorts from small to large
    # this is reversed because imshow plots the first row at the bottom, so it must be reversed to match the coordinate system
    eastings  = np.sort(mask_df['easting'].unique()) # Takes the Easting column --> removes duplicates = unique grid coords --> sorts from small to large
    n_north, n_east = len(northings), len(eastings) # Gets the number of unique northing and easting coordinates, which represent the dimensions of the grid

    # Reconstruct 2-D catchment mask (NaN = outside, 1 = inside): this is to identify which grid cells are inside the catchment and which are outside
    mask_2d = np.full((n_north, n_east), np.nan) # constructs a 2D array of shape (n_north, n_east) filled with NaN values, representing the catchment mask

    # Translate the real world coordinates (northings, eastings) into array indices (r, c) for the mask_2d array
    north_idx = {v: i for i, v in enumerate(northings)} # Creates a dictionary mapping each unique northing value (v) to its corresponding index (i)in the northings array
    east_idx  = {v: i for i, v in enumerate(eastings)} # Creates a dictionary mapping each unique easting value (v) to its corresponding index (i) in the eastings array

    # Populate the mask_2d array with 1.0 for points inside the catchment and NaN for points outside the catchment
    for _, row in mask_df.iterrows(): # For each row in the mask_df DataFrame, which contains the catchment mask data...
        r = north_idx[row['northing']] # translate the northing coordinate to the coresponding row index in the mask_2d array (e.g. if the northing is the 3rd unique northing value, then r = 2)
        c = east_idx[row['easting']]  # translate the easting coordinate to the corresponding column index in the mask_2d array (e.g. if the easting is the 5th unique easting value, then c = 4)
        if pd.notna(row['catchment_mask']): # Check if the catchment_mask value is not NaN (i.e. that this point is inside the catchment)
            mask_2d[r, c] = 1.0 # add 1.0 to the grid cell in the mask_2d array corresponding to this northing and easting coordinate, indicating that this point is inside the catchment

    # Pixel columns (pixel_000000, pixel_000001, …)
    pixel_cols = [c for c in flat_df.columns if c.startswith('pixel_')] # Creates a list of column names in the flat_df DataFrame that start with 'pixel_', which represent the rainfall values for each pixel in the grid
    T = len(flat_df) # Gets the number of time steps in the flat_df DataFrame, which is the number of rows in the DataFrame
    values = flat_df[pixel_cols].to_numpy().reshape(T, n_north, n_east) # Converts the rainfall values in the flat_df DataFrame to a NumPy array and reshapes it into a 3D array of shape (T, n_north, n_east), where T is the number of time steps, n_north is the number of unique northing coordinates, and n_east is the number of unique easting coordinates. This represents the rainfall values for each pixel in the grid at each time step.

    return flat_df, values, mask_2d, northings, eastings
    
    # Overall, this loads the rainfall data and converts it from a flat table to a 2D grid through time, and makes a catchment mask to identify which grid cells are inside the catchment and which are outside. 
    # The function returns the flattened rainfall DataFrame (colnames: pixel_000000, pixel_000001, ...), the 3D rainfall values array (T, n_north, n_east), 
    # the 2D catchment mask (mask_2d, containing 1.0 for inside and NaN for outside), and the unique northing and easting coordinates (northings, eastings, as arrays).


# ---This function draws a black contour around the catchment boundary on a given matplotlib Axes object.
# It takes in the Axes object, a 2D mask array indicating which pixels are inside the catchment, and the corresponding eastings and northings coordinates. 
# The function uses the contour method to draw a contour line at the level of 0.5, which effectively outlines the catchment area.
def draw_catchment_outline(
    ax: plt.Axes,
    mask_2d: np.ndarray,
    eastings: np.ndarray,
    northings: np.ndarray,
) -> None:
    """Draw a black contour around the catchment boundary."""
    inside = np.isfinite(mask_2d).astype(float)
    if inside.any():
        ax.contour(
            eastings, northings, inside,
            levels=[0.5],
            colors='black',
            linewidths=1.5,
        )

# -- Compute the mean rainfall over the catchment at each time step ------------------------------------------
def catchment_mean_rainfall(
    values: np.ndarray,
    mask_2d: np.ndarray,
) -> np.ndarray:
    """
    Compute the spatial mean of rainfall over the catchment at every time step.
    Pixels outside the catchment (NaN in mask_2d) are excluded.
    """
    inside = np.isfinite(mask_2d)          # shape (n_north, n_east): this identifies the catchment pixels (True) and the outside pixels (False)
    masked = np.where(inside, values, np.nan)  # (T, n_north, n_east):this keeps only the rainfall values within the catchment and replaces the outside values with NaN
    return np.nanmean(masked, axis=(1, 2)) # This takes the mean for the pixels in the catchment, and ignores those outside.
    # = a 1-D array of length T (the number of time steps), containing the mean rainfall over the catchment at each time step.

    
