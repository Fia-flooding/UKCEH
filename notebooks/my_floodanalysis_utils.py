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
    # The function returns the flattened rainfall DataFrame (colnames: pixel_000000, pixel_000001, ...), the 3D rainfall values array (T, n_north, n_east, rainfall values), 
    # the 2D catchment mask (mask_2d, containing 1.0 for inside and NaN for outside), and the unique northing and easting coordinates (northings, eastings, as arrays).


# ---This function draws a black contour around the catchment boundary on a given matplotlib Axes object.
# It takes in the Axes object, a 2D mask array indicating which pixels are inside the catchment, and the corresponding eastings and northings coordinates. 
# The function uses the contour method to draw a contour line at the level of 0.5, which effectively outlines the catchment area.
def draw_catchment_outline(
    ax: plt.Axes, # takes axis argument (shows where to plot the contour)
    mask_2d: np.ndarray, # takes a numpy array as the 2d mask --> what is a 2d mask?
    eastings: np.ndarray, # array of eastings
    northings: np.ndarray, # array of northings
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

    
# ----- Statistically identifying the flood event (start, peak and end) from a data set: ----
# Trying to write code/ functions to compute this:

# Calculating the variance across a moving window to identify where there is a sudden
# increase in variance, and therefore where there might be the start of a flood event
# this returns a variance series which is put into the next function along
def moving_variance(Q, window):
    return Q.rolling(window= window, min_periods = 1).var()

# Determine the flood threshold (fth) over which the variance must exceed for it to be counted as a 
# flood event, rather than just background flood flow
# the threshold in fischer et al., 2021 is defined as the mean of the moving variances + 0.25 *
# the standard deviation of these variances
def compute_fth(var_series, std_var_multiple = 2.5):
    mean_var = var_series.mean()
    std_var = var_series.std()
    return mean_var + std_var_multiple * std_var # threshold raised to 2.5 * std_var, so it only detects stronger variability signals


# Identify the flood event by taking in a series/ column from a dataframe containing the discharge flow
def find_events(Q, dvar=7, peak_flow_multiple = 5, time_slice_multiple = 10, end_idx_multiple = 0.15, min_time_slices = 5, flood_closeness_time_slices = 5): #dvar = 5 = stops the variance reacting too much to tiny wiggles
    
    """
    Q : discharge flow series
    dvar : ___
    peak_flow_multiple : defines how many time slices into the future, relative to the length of the ascending arm,
    should be searched for the peak threshold. Default is 5 * the length of the ascending arm
    time_slice_multiple : defines how short a flood rising event is before it is discarded as too short to be a true
    flood event. Default is 10 time slices (corresponding to discharge increaes for durations of 150 minutes in 15 min
    temporal resolution data). This should vary depending on the temporal resolution of the data.
    end_idx_multiple : defines how low the discharge volume should be, relative to the peak flow, to help determine the end
    of the flood event (one of two criteria). Default is 0.15 (i.e. flow should be 15% of the peak flow) but this can be varied
    depending on the catchment
    min_time_slices : defines how many time slices long theevent must be to be discarded as not a real event. Here, default is 5,
    so with a 15 min temporal resolution dataset flood events with rising limbs shorter than 75min long aren't counted as true
    flood events, to prevent false identification of many flood events
    flood_closeness_time_slices : if more than one peak is identified, this defines how much time can pass before they are treated
    as separate events, or until they are merged. The default here is 5 time slices, so if less than 75 mins (5 * 15 min res) passes 
    between two separate flood peaks, they are merged into the same flood event. 
    """
    # Convert the flow into a series
    Q = pd.Series(Q).reset_index(drop=True) 
    Q = Q.rolling(window = 3, center= True).mean().bfill().ffill()
    # This smooths the discharge right after converting to a series, to remove any noise nad stop variance from flickering too much

    # Lag-1 differences
    QD = Q.diff() # Produces a new DF calld QD (i.e chanage in discharge) which contains the
                    # difference between each row and the row before in the discharge df
    QD.iloc[0] = 0 # This sets the very first difference as 0 (as there has been no diff yet)

    # Moving variance
    var_series = moving_variance(Q, dvar) # returns a variance series containing the moving variances

    # Threshold
    threshold = compute_fth(var_series) # coputing the threshold based on this variance series

    # Active points
    active = var_series > threshold # identifies where the flood events start, as per our criteria
                                    # returning a series (of same length as the discharge series)
                                    # of bool values (True = active event! False = not an event)
    events = [] # sets up an empty list to store results, where results will each be stored as a dictionary
    i = 0 # start index = 0
    n = len(Q) # n is the length of the event's dataframe (i.e. the total number of timesteps)

    # this whole looping logic is looping through the whole timeseries and extracting one flood event at a time
    while i < n: # while we haven't cycled though the whole event yet...
        # Detect if we are inside a flood event:
        if active.iloc[i] and QD.iloc[i] > 0:# and Q.iloc[i] > 0.3 * Q.max(): # Look at the current row/ index in the 'active' series. 
                            # if this index is True, it'll enter this loop. If it isn't True,
                            # it will move onto the next time slice and see whether that one is True
            # Find start
            start = i # initial index for the start of the event. This is shifted backwards in time by this:
            # flow_threshold = Q.quantile(0.25)
            while start > 0 and QD.iloc[start] >0 : # while we aren't at the beginning of the dataset                                                 # AND the flow is still increasing ...
                start -= 1 #... hop back to the previous row to check whether this is still increasing.
                # this loop is expandingthe event backwards through time to capture the full rising limb
                # = overall, this only allows forward movement through time if the variance stays above the threshold    
                # it will stop increasing once the flow no longer increases, or once we reach teh start of the dataset
            
            # Ignore events after main peak decay
            if start > Q.idxmax() + 200:
                i += 1
                continue

            end_temp = i # Find temporary end OF THE HIGH VARIANCE REGION (which we take as what was the initial start)
            while end_temp < n and active.iloc[end_temp]: # while we are still between the rising limb and 
                                                        # and the temporary end 
                end_temp += 1

            # Identify the Peak of the flow:
            search_end = min(end_temp + int(peak_flow_multiple * (end_temp - start)), n)

            peak_idx = Q.iloc[start:search_end].idxmax()  # find the maximum discharge value
            # ... calculate the max discharge volume in the rising limb
            # this sums the volume of water across the rising limb = ~ the sum discharge between the start to the peak
            # ("search for a peak value within a region where we are confident there is something "event like" happening")
            # (this isn't even the full event, but rather just the active core of the event that we are looking at currently)
            if (peak_idx - start) < time_slice_multiple: # if this rising limb is too short (i.e. shorter than 150 minutes = ~ 2hours), ignore the event and move
                i += 1
                continue

            # -- only real events should reach this point -- #

            # End using volume balance
            rising_vol = Q.iloc[start:peak_idx].sum() # Calculate the volume of water discharged between the start of the event and the event peak
            cumulative = 0
            end = peak_idx # this is a safe default initialisation of the end index. i.e. “If we can’t find a better end, at least we have the peak”

            # Identifying the end of the event
            for j in range(peak_idx + 1, n): # start at the peak's index & move foward through time for the duration of the rest of the series
                cumulative += Q.iloc[j] # add that flow value to the cumulative (i.e. falling limb integral)
                if cumulative >= rising_vol and Q.iloc[j] < end_idx_multiple * Q.iloc[peak_idx]:# Repeat this until the descending limb vol equals or exceeds the rising limb vol...
                                                                                    #AND until the discharge is less than 15% of the peak flow.
                    end = j # then identify at which index this value is achieved...
                    break # and end the loop

            if (end - start) >= min_time_slices: # this removes events that are smaller than 5 * temp res (i.e. 75 min long), which are fake events probably
                events.append({ # add the indices to the events list of dictionaries
                "start": start,
                "peak": peak_idx,
                "end": end
            })

            i = end + 1 # Once the start, peak and end are found, it doesn't go onto the next row from the start row,
                        # but instead jumps straight to the index after the end of the event, so it can't double/ triple etc. count the same event!
        else:
            i += 1 # and if the current point isn't active, move onto the next one to check that one. 
        
        events_df = pd.DataFrame(events)
        
        # Events are merged post-processing to convert many tiny flood signals into one comprehensive flood signal
        if len(events_df) > 1:
            merged = [events_df.iloc[0].to_dict()] # this is the first detected event...

            for _, row in events_df.iloc[1:].iterrows(): #each subsequent flood event is then looked at...
                prev = merged[-1] # the current event and previous merged event are compared

                # the time gap between the events is checked. If the gap is small enough, they are close enough to be the same flood event
                if row['start'] - prev['end'] <= flood_closeness_time_slices:
                    prev['end'] = row['end'] # this extends the end of the event to cover both events
                    prev['peak'] = max(prev['peak'], row['peak']) # this updates the peak so the peak is still the highest value
                else:
                    merged.append(row.to_dict()) # but they are kept as separate events if they aren't close to each other. 

            events_df = pd.DataFrame(merged)


    return events_df
##################################################################

def get_catchment_coords(event_dict, nrfa_df):
    """
    Takes a dictionary of event metadata, and the nrfa peak flow dataframe
    (or another appropriate dataframe with centroid and gauage coordinates)

    Returns a list of the catchment IDs
    Returns a dataframe with the centroid and gauge coordinates of the catchments
    """

    # Extracting the IDs
    catchment_ID_list = []
    for dict in event_dict:
        catchment_ID_list.append(dict["catchment_id"])
    # Making the IDs integers
    for i, id in enumerate(catchment_ID_list):
        catchment_ID_list[i] = int(id)

    # Filtering out for the coordinates only:
    nrfa_coordinates = nrfa_df[['Station', 'Easting', 'Northing', 'CEasting', 'CNorthing']]

    # Checking whether this ID is in this dataset:
    for id in catchment_ID_list:
        if id not in nrfa_coordinates['Station'].values:
            print(f"WARNING:\nCatchment ID {id} is not in the NRFA Peak Flow\n" \
            "dataset; Search for its coordinates elsewhere, and concatenate these" \
            "onto the dataframe manually")

    # Filtering the events of these catchment IDs out of the dataset:
    nrfa_coordinates = nrfa_coordinates[
        nrfa_coordinates['Station'].isin(catchment_ID_list)
        ]
    
    return catchment_ID_list, nrfa_coordinates


################################################################
# Calculating the ACW rotation for the catchments and maps
def calc_ACW_rotation(gauge_centroid_coord_df):
    """
    Calculates the ACW rotation needed for the centroid to sit over the gauage location
    Takes a df with the gauge and centroid coordinates 
    Returns a dataframe with the acw rotation
    """
    dx = gauge_centroid_coord_df['Easting'] - gauge_centroid_coord_df['CEasting']
    dy = gauge_centroid_coord_df['Northing'] - gauge_centroid_coord_df['CNorthing']

    # Calculate bearing from the vertical (= gives the clockwise angle from North)
    bearing = np.degrees(np.arctan2(dx, dy))

    # Adding these to the dataframe:
    gauge_centroid_coord_df['ACW_rotation_required'] = bearing + 180

    # Checking that this worked:
    rot_centroid_x = []
    rot_centroid_y = []

    rot_gauge_x = []
    rot_gauge_y = []

    for _, row in gauge_centroid_coord_df.iterrows():

        theta = np.radians(row['ACW_rotation_required'])

        # Vector from centroid to gauge
        dx = row['Easting'] - row['CEasting']
        dy = row['Northing'] - row['CNorthing']

        # Rotate vector
        dx_rot = dx*np.cos(theta) - dy*np.sin(theta)
        dy_rot = dx*np.sin(theta) + dy*np.cos(theta)

        # Put centroid at origin
        rot_centroid_x.append(0)
        rot_centroid_y.append(0)

        rot_gauge_x.append(dx_rot)
        rot_gauge_y.append(dy_rot)

    print("Maximum horizontal offset:",
        np.max(np.abs(rot_gauge_x)))

    print("Number of gauges below centroid:",
        np.sum(np.array(rot_gauge_y) < 0),
        "out of",
        len(rot_gauge_y))
    
    if np.max(np.abs(rot_gauge_x)) > 1: 
        print("Maximum horizontal offset is too large: centroid \nand gauge aren't over eachother")
    
    if np.sum(np.array(rot_gauge_y) < 0) < len(rot_gauge_y):
        print("Not all rotation angles are correct")


    return gauge_centroid_coord_df
##################################################################
