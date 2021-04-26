# Copyright 2021 Don McClimans
# This program is licensed under the MIT license. See the License.txt file for details

# noinspection PyUnresolvedReferences
from typing import List, Set, Dict, Tuple, Optional, Union
import gpxpy
import os
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import seaborn as sns
import pandas
from datetime import datetime, timezone
import math
import argparse
import numpy as np
import time
from lxml import etree


def meters_to_user_units_string(meters: float, units: str) -> str:
    """Convert a meters value to user units, and format as a string"""
    if units == 'english':
        return format(meters * 3.2808, '0.1f')
    else:
        return format(meters, '0.2f')


def meters_to_user_units(meters: float, units: str) -> float:
    """Convert a meters value to user units"""
    if units == 'english':
        return meters * 3.2808
    else:
        return meters


def user_units_to_meters(value: float, units: str) -> float:
    """Convert a user units value to meters"""
    if units == 'english':
        return value / 3.2808
    else:
        return value


def meters_to_user_units_scale_factor(units: str) -> float:
    """
    Return the scale factor to convert meters to user units.
    Multiply the meters value by the scale factor to get user units

    Args:
        units:  String containing 'english' or 'meters'

    Returns:    Scale factor
    """
    if units == 'english':
        return 3.2808
    else:
        return 1.0


def ceg_segment(segment: gpxpy.gpx.GPXTrackSegment, threshold: float, direction: int) -> Optional[float]:
    """
    Returns the CEG (cumulative elevation gain) or CEL (cumulative elevation loss) of the GPXTrackPoints
    in a GPXTrackSegment.

    Args:
        segment:    The GPXTrackSegment to analyze
        threshold:  The threshold distance (in meters)
        direction:  The direction to analyze. Use +1 for CEG and -1 for CEL.

    Returns:
        The CEG (or CEL), or None if segment does not have elevations (less than 75% of points have
        associated elevations).
    """
    if not segment.has_elevations:
        return None

    previous_good: Optional[float] = None
    elevation_sum: float = 0.0
    elevation_gain: float = 0.0
    for point in segment.points:
        if previous_good is None:
            previous_good = point.elevation
        elif point.elevation is not None:
            elevation_gain = point.elevation - previous_good
            if abs(elevation_gain) >= threshold:
                if (((direction >= 0) and (elevation_gain > 0)) or
                        ((direction < 0) and (elevation_gain < 0))):
                    elevation_sum += elevation_gain
                    elevation_gain = 0
                previous_good = point.elevation

    # The last numeric cell of the range is always considered a "good" data point.
    # Add it now. If the last point was already considered good and added,
    # elevation_gain will be zero.
    if (((direction >= 0) and (elevation_gain > 0)) or
            ((direction < 0) and (elevation_gain < 0))):
        elevation_sum += elevation_gain

    return elevation_sum


def ceg_track(track: gpxpy.gpx.GPXTrack, threshold: float, direction: int) -> Optional[float]:
    """
    Returns the CEG (cumulative elevation gain) or CEL (cumulative elevation loss) of the GPXTrackSegments
    in a GPXTrack.

    Does not count gains and losses that are less than the threshold.
    Parameters:
    track:      The GPXTrack to analyze
    threshold:  The threshold distance (in meters)
    direction:  The direction to analyze. Use +1 for CEG and -1 for CEL.
    """
    if not track.has_elevations:
        return None
    elevation_sum: float = 0
    for segment in track.segments:
        elevation_sum += ceg_segment(segment, threshold, direction) or 0.0

    return elevation_sum


def ceg(gpx: gpxpy.gpx.GPX, threshold: float, direction: int) -> Optional[float]:
    """
    Returns the CEG (cumulative elevation gain) or CEL (cumulative elevation loss) of the GPXTrackSegments
    in a GPXTrack.

    Does not count gains and losses that are less than the threshold.
    Parameters:
        gpx:        The GPX object  to analyze
        threshold:  The threshold distance (in meters)
        direction:  The direction to analyze. Use +1 for CEG and -1 for CEL.
    """
    if not gpx.has_elevations:
        return None

    elevation_sum: float = 0.0
    for track in gpx.tracks:
        elevation_sum += ceg_track(track, threshold, direction) or 0.0

    return elevation_sum


def filter_elevation2(gpx: gpxpy.gpx, kernel: tuple) -> None:
    size = len(kernel)
    if size % 2 == 0:
        raise RuntimeError('Elevation filter kernel size is not odd number')

    # Convolve the segment elevations with the filter kernel
    # For points that are closer to the end than the half-size, don't
    # do any convolution
    # Todo: Use part of the kernel for points near the end-point
    kernel_sum: float = sum(kernel)
    if kernel_sum == 0.0:
        raise RuntimeError('Elevation filter kernel sum is zero')
    half_size = (size - 1) // 2
    for track in gpx.tracks:
        for segment in track.segments:
            new_elevations: List[float] = []
            for i in range(len(segment.points)):
                if ((i - half_size) < 0) or ((i + half_size) >= len(segment.points)):
                    new_elevations.append(segment.points[i].elevation)
                else:
                    weighted_sum: Optional[float] = 0.0
                    for j in range(size):
                        if segment.points[i - half_size + j].elevation is None:
                            weighted_sum = None
                            break
                        weighted_sum += kernel[j] * segment.points[i - half_size + j].elevation
                    if weighted_sum is not None:
                        new_elevations.append(weighted_sum / kernel_sum)
                    else:
                        new_elevations.append(segment.points[i].elevation)
            for i in range(len(segment.points)):
                segment.points[i].elevation = new_elevations[i]


def filter_elevation(gpx: gpxpy.gpx, filter_method: str) -> None:
    filter_id = filter_method[0:1].lower()
    if filter_id == '0':
        return

    if filter_id == 'a':
        filter_elevation2(gpx, (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0))
    elif filter_id == 'b':
        filter_elevation2(gpx, (1.0 / 5.0, 1.0 / 5.0, 1.0 / 5.0, 1.0 / 5.0, 1.0 / 5.0))
    elif filter_id == 'c':
        filter_elevation2(gpx, (1.0 / 7.0, 1.0 / 7.0, 1.0 / 7.0, 1.0 / 7.0, 1.0 / 7.0, 1.0 / 7.0, 1.0 / 7.0))
    elif filter_id == 'd':
        filter_elevation2(gpx, (0.3, 0.4, 0.3))
    # elif filter_id == 'e':
    #     filter_elevation2(gpx, (0.4, 0.2, 0.4))
    # elif filter_id == 'f':
    #     gpx.smooth()


def write_gpx_file(gpx: gpxpy.gpx, input_filename: str, suffix: str) -> None:
    if not suffix or not suffix.strip():
        return

    # Add the suffix to each track name
    for track in gpx.tracks:
        track.name += suffix

    # Add the suffix to the filename.
    root, ext = os.path.splitext(input_filename)
    output_filename = root + suffix + ext

    # Fix up xml to be basecamp compatible.
    # Garmin basecamp has a bug that gives an "unknown" error when trying to open a file that has a
    # <bounds ...> start-tag with a </bounds> end-tag. However basecamp works fine with a empty-element
    # tag <bounds ... />.
    # Here I convert the start-tag and end-tag form (<bounds... ></bounds>) to a empty-element tag form
    # (<bounds.../>).
    xml = gpx.to_xml()
    pos1 = xml.find('<bounds ')
    if pos1 >= 0:
        pos2 = xml.find('>', pos1)
        if pos2 > pos1 and xml[pos1 - 1:pos1] != '/':
            pos3 = xml.find('</bounds>', pos2)
            if pos3 > pos2:
                xml = xml[:pos2] + '/>' + xml[pos3 + 9:]

    # Write the file.
    with (open(output_filename, 'w')) as output_file:
        output_file.write(xml)


def segment_to_elevation_list(segment: gpxpy.gpx.GPXTrackSegment) -> List[Optional[float]]:
    """Converts the elevation values in a segment to a list.

    Args:
        segment:    The track segment to convert

    Returns:
        A list of Optional[Float] values which are the elevations.
    """
    # Create a list of data. Each element is an elevation.
    data = []
    for point in segment.points:
        data.append(point.elevation)
    return data


def segment_to_time_list(segment: gpxpy.gpx.GPXTrackSegment) -> List[datetime]:
    """Converts the time values in a segment to a list.

    Args:
        segment:    The track segment to convert

    Returns:
        A list of times, with the timezones set to UTC.
    """
    # Create a list of data. Each element is a datetime object. The datetime object in the
    # segment.points[i].time has a timezone of SimpleTZ, which isn't hashable and causes any
    # conversion to a pandas dataframe to fail. So we first convert to a list and fix the timezone
    # to be UTC.
    data = []
    for point in segment.points:
        data.append(point.time.replace(tzinfo=timezone.utc))
    return data


def segment_to_time_segment_legend_list(segment: gpxpy.gpx.GPXTrackSegment, legend: str) -> \
        List[List[Union[datetime, Optional[float], str]]]:
    """Converts the time and elevation values in a segment to a list.

    Args:
        segment:    The track segment to convert
        legend:     String that will become the identifier (final value in return values)

    Returns:
        A of times, with the timezones set to UTC.
    """
    # Create a list of data. Each element is a datetime object. The datetime object in the
    # segment.points[i].time has a timezone of SimpleTZ, which isn't hashable and causes any
    # conversion to a pandas dataframe to fail. So we first convert to a list and fix the timezone
    # to be UTC.
    data = []
    for point in segment.points:
        data.append([point.time.replace(tzinfo=timezone.utc), point.elevation, legend])
    return data


def plot_elevations(input_gpx: gpxpy.gpx, input_filename: str, pre_difference_gpx: gpxpy.gpx,
                    difference_gpx: gpxpy.gpx, output_gpx: gpxpy.gpx,
                    args: argparse.Namespace, is_interactive: bool) -> None:
    """
    Plot the elevations from a gpx object.

    There will be a separate plot for each segment in each track.

    Args:
        input_gpx:          The input gpx object
        input_filename:     The filename of the input file. Used  to rename the plot file
        pre_difference_gpx: The gpx object that corresponds to the data before subtracting the difference file. This
                            is the data after filtering.
        difference_gpx:     The gpx object that corresponds to the difference file. Can be None
        output_gpx:          The gpx object after filtering and subtracting the difference
        args:               The args parse structure that gives the arguments passed to the command line or
                            set in the GUI. The values used are:
            show_plot:      True to show interactive plot
            plot_suffix:    Suffix and extension to add to base filename for plot file. If empty
                            string, no plot file.
            plot_input:     Add input elevations to plot
            plot_before_difference: Add before-difference elevations to the plot
            plot_difference_file:   Add difference file elevations to plot
            plot_output     Add output elevations (after filter and difference) to the plot
            units:          The units string, either 'english' or 'metric'
        is_interactive:     True if run under gui, false if run from command line.
                            If false, and plot_elev is true, plots block further processing
                            so you can view and interact with the plot.

    Returns: None

    """
    # You must be either showing a plot or saving a plot file.
    if not args.show_plot and not args.plot_suffix:
        return
    # You must turn on one of the plot options
    if not args.plot_input and not args.plot_before_difference and not args.plot_difference_file and \
            not args.plot_output:
        return

    for track_idx in range(len(input_gpx.tracks)):
        for segment_idx in range(len(input_gpx.tracks[track_idx].segments)):
            # Create the data list. This is a long format data structure (in pandas sense of long vs wide).
            data = []
            num_points = len(input_gpx.tracks[track_idx].segments[segment_idx].points)

            # plot the input if requested
            if args.plot_input:
                data.extend(segment_to_time_segment_legend_list(input_gpx.tracks[track_idx].segments[segment_idx],
                                                                'input'))

            # plot the pre-difference if requested.
            if args.plot_before_difference \
                    and pre_difference_gpx is not None \
                    and track_idx < len(pre_difference_gpx.tracks) \
                    and segment_idx < len(pre_difference_gpx.tracks[track_idx].segments) \
                    and len(pre_difference_gpx.tracks[track_idx].segments[segment_idx].points) == num_points:
                data.extend(segment_to_time_segment_legend_list(
                    pre_difference_gpx.tracks[track_idx].segments[segment_idx], 'before diff'))

            # plot the difference file if requested.
            if args.plot_difference_file \
                    and difference_gpx is not None \
                    and track_idx < len(difference_gpx.tracks) \
                    and segment_idx < len(difference_gpx.tracks[track_idx].segments):
                data.extend(segment_to_time_segment_legend_list(
                    difference_gpx.tracks[track_idx].segments[segment_idx], 'diff file'))

            # plot the output if requested.
            if args.plot_output \
                    and output_gpx is not None \
                    and track_idx < len(output_gpx.tracks) \
                    and segment_idx < len(output_gpx.tracks[track_idx].segments) \
                    and len(output_gpx.tracks[track_idx].segments[segment_idx].points) == num_points:
                data.extend(segment_to_time_segment_legend_list(output_gpx.tracks[track_idx].segments[segment_idx],
                                                                'output'))

            # Create the dataframe.
            sensor_df = pandas.DataFrame(data, columns=['time', 'elevation', 'legend'])
            # sensor_df.info(verbose=True)
            # print(sensor_df)

            # Convert to user units
            sensor_df['elevation'] *= meters_to_user_units_scale_factor(args.units)

            # Plot the dataFrame.
            fig, axes = plt.subplots()
            sns.lineplot(data=sensor_df, x='time', y='elevation', hue='legend')

            # Set the axis labels.
            # Set the x axis to show HH:MM
            plt.xlabel('Time (UTC)')
            plt.ylabel(f'Elevation ({"feet" if args.units == "english" else "meters"})')
            plt.xticks(rotation=30)
            time_format = mdates.DateFormatter('%H:%M')
            axes.xaxis.set_major_formatter(time_format)

            # Put the legend at the bottom of the chart, and make it horizontal.
            plt.legend(ncol=5, loc="lower center", bbox_to_anchor=(0.5, -0.3))

            # tight_layout rearranges everything so it fits
            plt.tight_layout()

            # Save the file if requested
            if args.plot_suffix:
                # Add the suffix to the filename and save the file.
                root, ext = os.path.splitext(input_filename)
                plot_filename: str = root + args.plot_suffix
                if plot_filename == input_filename:
                    print('Plot file cannot overwrite input file. Plot file not written')
                else:
                    plt.savefig(plot_filename, dpi=200)

            # Show the plot if requested
            if args.show_plot:
                plt.show(block=not is_interactive)


def print_stats(gpx: gpxpy.gpx, args: argparse.Namespace, is_interactive: bool) -> None:
    for track in gpx.tracks:
        print(f'Track: {track.name}')
        # if args.gpxpy_up_down:
        #     uphill, downhill = track.get_uphill_downhill()
        #     print(f'  gpxpy uphill = {meters_to_user_units_string(uphill, args.units)}'
        #           f' downhill = {meters_to_user_units_string(downhill, args.units)}')

        ceg_list: List[List[float, float]] = []
        if args.ceg_threshold >= 0:
            if not args.all_ceg:
                threshold = float(args.ceg_threshold)
                threshold_meters = user_units_to_meters(float(args.ceg_threshold), args.units)
                print(f'  CEG({threshold}) = '
                      f'{meters_to_user_units_string(ceg_track(track, threshold_meters, 1), args.units)}'
                      f'  CEL({threshold}) = '
                      f'{meters_to_user_units_string(ceg_track(track, threshold_meters, -1), args.units)}'
                      )
            else:
                start = 0
                stop = math.ceil(float(args.ceg_threshold)) + 1
                for threshold in range(start, stop):
                    threshold_meters = user_units_to_meters(float(threshold), args.units)
                    ceg_meters = ceg_track(track, threshold_meters, 1)
                    cel_meters = ceg_track(track, threshold_meters, -1)
                    ceg_list.append([threshold, meters_to_user_units(ceg_meters, args.units)])
                    print(f'  CEG({threshold}) = {meters_to_user_units_string(ceg_meters, args.units)}'
                          f'  CEL({threshold}) = {meters_to_user_units_string(cel_meters, args.units)}'
                          )

        if args.plot_ceg and len(ceg_list) > 1:
            # Create the dataframe.
            sensor_df = pandas.DataFrame(ceg_list, columns=['threshold', 'CEG'])
            # sensor_df.info(verbose=True)
            # print(sensor_df)

            fig, ax = plt.subplots()
            sns.lineplot(data=sensor_df, x='threshold', y='CEG', ax=ax)

            # Set the axis labels.
            plt.xlabel(f'Threshold ({"feet" if args.units == "english" else "meters"})')
            plt.ylabel(f'CEG ({"feet" if args.units == "english" else "meters"})')
            plt.title('Cumulative Elevation Gain')

            # tight_layout rearranges everything so it fits
            plt.tight_layout()

            # Show the plot if requested
            plt.show(block=not is_interactive)

        # if args.gpxpy_up_down or (args.ceg_threshold >= 0):
        if args.ceg_threshold >= 0:
            min_elevation, max_elevation = track.get_elevation_extremes()
            if min_elevation is None:
                min_elevation = 0.0
            if max_elevation is None:
                max_elevation = 0.0
            print(f'  Min: {meters_to_user_units_string(min_elevation, args.units)}'
                  f'  Max: {meters_to_user_units_string(max_elevation, args.units)}'
                  f'  Max-Min: {meters_to_user_units_string(max_elevation-min_elevation, args.units)}')


def subtract_difference(gpx: gpxpy.gpx, difference_gpx: gpxpy.gpx, difference_filename: str):
    if not gpx or not difference_gpx:
        return
    start = time.time()

    for track_idx in range(len(gpx.tracks)):
        track = gpx.tracks[track_idx]
        if track_idx >= len(difference_gpx.tracks):
            print(f'Missing track # {track_idx} in {difference_filename}')
        else:
            difference_track = difference_gpx.tracks[track_idx]
            for segment_idx in range(len(track.segments)):
                if segment_idx >= len(difference_track.segments):
                    print(f'Missing track # {track_idx} segment # {segment_idx} in {difference_filename}')
                else:
                    segment = track.segments[segment_idx]

                    # The difference gpx file have different time-stamps from the input gpx file.
                    # So we will search and interpolate in the difference_gpx file.
                    # Convert difference_gpx to numpy arrays to make this efficient.
                    difference_segment = difference_track.segments[segment_idx]
                    difference_timestamps, difference_elevations = \
                        convert_segment_points_to_arrays(difference_segment)

                    for i in range(len(segment.points)):
                        if segment.points[i].time is None:
                            segment.points[i].elevation = None
                        elif (i < len(difference_segment.points)) and \
                                (segment.points[i].time.timestamp() == difference_timestamps[i]):
                            # The difference file timestamp matches the gpx timestamp. Subtract on a
                            # point by point basis.
                            #  Note if there is a missing timestamp or elevation, they have been
                            #  removed from the difference arrays. In that case we will apply
                            # the interpolation method. You could correct for this by converting
                            # the segment elevations to ndarrays as well, but they you have to figure
                            # out the index in the segment elevation to update -- not worth the trouble.
                            # In my testing this optimization reduced the time from 23 msec for the
                            # interpolation method to 8 ms for this straight lookup.
                            if (difference_segment.points[i].elevation is not None) and \
                                    (segment.points[i].elevation is not None):
                                segment.points[i].elevation -= difference_segment.points[i].elevation
                            else:
                                segment.points[i].elevation = None
                        else:
                            # The difference file uses different timestamps than the gpx file.
                            # Interpolate in the difference file to get the elevation at the time matching
                            # the gpx file, and subtract it.
                            if segment.points[i].elevation is not None:
                                interpolated_difference = np.interp(segment.points[i].time.timestamp(),
                                                                    difference_timestamps, difference_elevations,
                                                                    left=np.nan, right=np.nan)
                                if np.isnan(interpolated_difference):
                                    segment.points[i].elevation = None
                                else:
                                    segment.points[i].elevation -= interpolated_difference

    end = time.time()
    # print(f'subtract_difference elapsed time {end - start}')


def read_csv_pressure_file(csv_filename: str, is_interactive: bool) -> Optional[pandas.DataFrame]:
    """
    Read csv file with pressure data.
    Works with data files from tempo disc device.

    Args:
        csv_filename:   The filename of the csv (tempo disc) file.
                        If None or empty string, not an error, returns None
        is_interactive: True if running under gui.

    Returns:

    """
    # Read the Sensor file if one is specified
    if not csv_filename:
        return None

    try:
        sensor_dfheading = pandas.read_csv(csv_filename, nrows=2)
        sensor_df = pandas.read_csv(csv_filename, skiprows=2, parse_dates=['date'])
    except (IOError, ValueError, Exception) as exc:
        print(f'Cannot read sensor file:\n    {csv_filename}\nError: {str(exc)}')
        return None

    if sensor_df is None or sensor_df.empty or sensor_dfheading is None or sensor_dfheading.empty:
        return None

    # sensor_dfheading contains the first two lines of the file. This is a heading line plus one data line.
    # We use it to determine if the file contains temperatures in Fahrenheit or Celsius
    # If Fahrenheit, then all temps are converted to Celsius
    if ('Temp_Units' in sensor_dfheading.columns) and (sensor_dfheading['Temp_Units'][0] == "Fahrenheit"):
        sensor_df['temperature'] = (sensor_df['temperature'] - 32.0) * 5.0 / 9.0

    # Parse the datetime field. Use exact=False to ignore the "(Mountain Standard Time)" text at the end.
    # The format string (strptime) does not allow you to skip characters.
    sensor_df['date'] = \
        pandas.to_datetime(sensor_df['date'], format='%a %b %d %Y %H:%M:%S GMT%z', exact=False)

    # Debugging code to show pressure plot.
    if False:  # set True to show pressure plot (for debugging)
        sensor_df.info(verbose=True)
        print(sensor_df)
        print(type(sensor_df['date'][0]))

        # For debugging, look at the pressure chart
        grid: sns.axisgrid.FacetGrid = sns.relplot(data=sensor_df, kind='line', x='date', y='pressure')
        grid.set_xticklabels(rotation=30)
        grid.set_xlabels('Time (MM-DD HH UTC)')
        grid.set_ylabels('hPa')
        plt.title('Pressure')
        plt.tight_layout()
        plt.show(block=not is_interactive)

    return sensor_df


def pressure_to_elevation(pressure: float, p0: float) -> float:
    """
    Calculate the elevation for a given pressure.
    Args:
        pressure:   The pressure, in hPa (mbars)
        p0:         P0, the pressure at sea level.

    Returns: Elevation in meters

    """
    return (1-((pressure/p0)**0.190284))*145366.45*0.3048


def gpx_pressures_to_elevations(elevations: np.ndarray, pressures: np.ndarray, p0: float):
    """
    Convert an array of pressures to elevations
    Args:
        elevations: Array of returned elevations.
        pressures:  Array of pressure values
        p0:         The calibration p0

    Returns:

    """
    for idx, pressure in enumerate(pressures):
        if np.isnan(pressure):
            elevations[idx] = np.nan
        else:
            elevations[idx] = round(pressure_to_elevation(pressure, p0), 2)


def calculate_p0(pressure: float, elevation: float) -> float:
    """
    Calculate P0 from pressure and desired elevation
    Args:
        pressure:   The pressure at this elevation
        elevation:  The desired elevation

    Returns: The calibrated P0
    """
    p0: float = pressure / ((1 - elevation / 0.3048 / 145366.45) ** (1/0.190284))
    return p0


def get_point_data(gpx_timestamps: np.ndarray, gpx_elevations: np.ndarray,
                   sensor_timestamps: np.ndarray, sensor_pressures: np.ndarray,
                   sensor_temperatures: np.ndarray) \
                   -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Create a list of pressures based on the pressure data.
    Args:
        gpx_timestamps:         Array of datetime.timestamps (floats) from the gpx files
        gpx_elevations:         Array of gpx elevations.
        sensor_timestamps:      Array of datetime.timestamps (floats) from the barometric sensor
        sensor_pressures:       Array of pressures from the barometric sensor
        sensor_temperatures:    Array of temperatures from the sensor

        The arrays must not have any None values, they are only float values.
        The sensor_timestamps and sensor_pressures arrays must be the same size

    Returns:
        Array of gpx_timestamps: This will be the input parameter gpx_timestamps, except
                                 with all points removed that could not have pressures
                                 calculated. This happens when the gpx_timestamp is outside
                                 the barometer timestamps.
        Array of gpx_elevations: The gpx_elevations input array, except with all points
                                 removed that could not have pressures calculated. This
                                 happens when the gpx_timestamp is outside the barometer
                                 timestamps.
        Array of gpx_pressures:  The pressure values associated with the gpx_timestamps.
    """
    start = time.time()

    # Slice the sensor timestamp, pressure, and temperature arrays to only include the data
    # that applies to the times in segment_point_times. This makes the lookup run faster.
    # It's not really significant now that I use ndarrays instead of lists and pandas
    # DataFrames and series, but it's easy to do and does speed things up a little.
    sensor_len = len(sensor_timestamps)
    start_time = gpx_timestamps[0]
    end_time = gpx_timestamps[len(gpx_timestamps)-1]
    start_search = np.searchsorted(sensor_timestamps, start_time, side='left')-1
    end_search = np.searchsorted(sensor_timestamps, end_time, side='right')+1
    if start_search < 0:
        start_search = 0
    if end_search > sensor_len:
        end_search = sensor_len
    sensor_timestamp_array = sensor_timestamps[start_search:end_search]
    sensor_pressure_array = sensor_pressures[start_search:end_search]
    sensor_temperature_array = sensor_temperatures[start_search:end_search]

    # For each point in segment_point_times, find the two surrounding times in the
    # timestamp array. Then use the corresponding values in the pressure array
    # to interpolate the pressure at that time.
    # Do the same with the temperature data
    gpx_new_timestamps: np.ndarray = np.zeros_like(gpx_timestamps)
    gpx_new_elevations: np.ndarray = np.zeros_like(gpx_elevations)
    gpx_pressures: np.ndarray = np.zeros_like(gpx_timestamps)
    gpx_temperatures: np.ndarray = np.zeros_like(gpx_timestamps)
    new_idx: int = 0
    for idx, point_time in enumerate(gpx_timestamps):
        # Interpolate in the pressure_array.
        pressure = np.interp(point_time, sensor_timestamp_array, sensor_pressure_array,
                             left=np.nan, right=np.nan)
        if not np.isnan(pressure):
            # Convert pressure to uncalibrated elevation
            gpx_pressures[new_idx] = pressure
            gpx_new_timestamps[new_idx] = gpx_timestamps[idx]
            gpx_new_elevations[new_idx] = gpx_elevations[idx]
            temperature = np.interp(point_time, sensor_timestamp_array, sensor_temperature_array,
                                    left=np.nan, right=np.nan)
            if np.isnan(temperature):
                if new_idx > 0:
                    temperature = gpx_temperatures[new_idx-1]
                else:
                    temperature = 0.0
            gpx_temperatures[new_idx] = temperature
            new_idx += 1

    end = time.time()
    # print(f'get_point_data elapsed time {end - start}')

    # If any timestamps were outside the barometer timestamps, warn the user
    if new_idx < len(gpx_timestamps):
        print(f'{100.0*new_idx/len(gpx_timestamps)}% of points in the segment were outside the barometer timestamps')

    # Return a slice (view) of the timestamps, elevations, and pressures.
    # This is usually the same size as the input arrays, but an be smaller if
    # the gpx_timestamp is outside the sensor_timestamps
    return gpx_new_timestamps[:new_idx], gpx_new_elevations[:new_idx], gpx_pressures[:new_idx], \
           gpx_temperatures[:new_idx]


# Define constants the control calibration.
# If needed, these could be come functions and be settable by command line parameters
# Units are seconds (same which match timestamps).

# Skip initial data interval -- 5 minutes
# Some (many) gpx units start up with wildly wrong elevation data.
skip_initial_interval: float = 5.0 * 60.0

# Beginning average interval -- 5 minutes
# When using method A, skip over skip_initial_interval, then average over
# beginning_average_interval
beginning_average_interval: float = 5.0 * 60.0

# Section length - 1 hour
# Divide the gpx segment into sections of 1 about 1 hour
section_interval: float = 60.0 * 60.0


def add_constant_elevation(gpx_pressure_elevations: np.ndarray, gpx_pressures: np.ndarray, offset: float) -> None:
    """
    Add a constant height adjustment to all elevations in pressure_elevations

    Args:
        gpx_pressure_elevations:
        gpx_pressures
        offset:

    Returns:

    """
    if len(gpx_pressure_elevations) < 1:
        return

    # Find the pressure and elevation at the midpoint. Adjust the elevation by the offset.
    # Calculate a new P0 based on the pressure and desired elevation at that point.
    mid_idx: int = len(gpx_pressure_elevations)//2
    p0 = calculate_p0(gpx_pressures[mid_idx], gpx_pressure_elevations[mid_idx] + offset)

    # Recalculate the pressure_elevations using the new p0.
    # ReCalculate new (calibrated) elevations from the pressure data.
    gpx_pressures_to_elevations(gpx_pressure_elevations, gpx_pressures, p0)


def find_closest_timestamp_idx(gpx_timestamps: np.ndarray, search_timestamp: float) -> int:
    """
    Finds the index of the time in gpx_timestamps that is closest to search_timestamp.

    Args:
        gpx_timestamps:     Array of timestamps to search
        search_timestamp:   Time to search for

    Returns:
        Index of element in gpx_timestamps that is closest to the search_timestamp.
        Will be between 0 and len(gpx_timestamps)-1
    """
    idx: int = np.searchsorted(gpx_timestamps, search_timestamp)
    if idx <= 0:
        # search_timestamp is before start. return 0.
        return idx
    if idx >= len(gpx_timestamps):
        # search_timestamp is beyond end. Return last point
        return len(gpx_timestamps)-1
    # Search_timestamp is between idx-1 and idx. Find the closest
    if abs(gpx_timestamps[idx-1]-search_timestamp) < abs(gpx_timestamps[idx]-search_timestamp):
        return idx-1
    else:
        return idx


def calibrate_elevations2(gpx_timestamps: np.ndarray,
                          gpx_pressures: np.ndarray, gpx_pressure_elevations: np.ndarray,
                          elevation_differences: np.ndarray.view,
                          ) -> None:
    """
    # Calibration Method C
    # Break the file into sections.
    # Calculate the average elevation_difference over a section_interval, which gives you the offset
    # to apply at the midpoint of the section. Calibrate between endpoints by interpolating the offsets
    # between the endpoints. On the edges, continue calibrating by extrapolating the first and last
    # section calibration values.

    Args:
        gpx_timestamps:
        gpx_pressures:
        gpx_pressure_elevations:
        elevation_differences:

    Returns:

    """
    segment_count: int = len(gpx_timestamps)
    start_time: float = gpx_timestamps[0]
    stop_time: float = gpx_timestamps[segment_count-1]
    segment_time: float = stop_time - start_time

    # Description of the algorithm
    #
    # Calculate the number of sections (section_count) by dividing the segment time, less the
    # skip_initial_interval, by the segment time. Set a minimum of 2 segments.
    #
    # Then calculate the section midpoints and the section endpoints (in seconds).
    # The first section starts at start_time+skip_initial_interval, and the last section ends
    # at stop_time. The section endpoints and midpoints are times -- they do not necessarily
    # correspond to a specific data point in the segment.
    #
    # Calculate the offset for each segment by averaging over the segment. At the midpoint, calculate
    # the P0 that is required to produce that offset.
    #
    # Recompute all elevations based on a new P0. Between section mid-points use a P0
    # calculated from the P0 at the two midpoints, interpolating (based on time not array indexes).
    # At the beginning (and end) of the data, extrapolate the P0 based on the two section mid-points
    # at the beginning (and end) of the section point list.
    #
    # For example, say the segment is 1:55 (h:mm) long, you have a skip_initial_interval of 5 minutes,
    # and a section_interval of 60 minutes.
    # Subtract the skip_interval of 5 minutes and divide by 60 minutes section_interval, rounds to
    # 2 segments. The section endpoints are [0:05, 1:00, 1:55] and the midpoints are [0:27.5, 1:27.5].
    #
    # Here is a table showing the minimum and maximum length of the sections as a function of the
    # segment length (less the skip_initial_interval). This table assumes the same 60 minute
    # section_interval and 5 minute skip_initial_interval:
    #             Just under         Just over
    # Seg len   num secs sec len | num secs sec len
    # 0:60          2      0:30        2      0:30
    # 1:30          2      0:45        2      0:45
    # 2:30          2      1:15        3      0:50
    # 3:30          3      1:10        4      0:52.5
    #
    # As you continue to increase the segment length, the section length approaches 1:00.
    # For segments shorter than 1:30, the calculated number of sections is 1, but with only
    # 1 section we can't calculate a slope, so we force the number of segments to 2. That forces
    # the section length down. We don't call this function for segments shorter than 60 minutes,
    # so our minimum section time is 30 minutes.

    # Calculate the number of sections, and the section length.
    section_count: int = round((segment_time - skip_initial_interval) / section_interval)
    section_count = max(section_count, 2)

    # Calculate the section end-points and mid-points
    section_endpoint_times: np.ndarray
    section_time: float
    section_endpoint_times, section_time = np.linspace(start_time+skip_initial_interval, stop_time,
                                                       num=(section_count+1), retstep=True)
    section_endpoints: np.ndarray = np.zeros(section_count+1, dtype=int)
    section_midpoints: np.ndarray = np.zeros(section_count, dtype=int)
    for i in range(section_count+1):
        idx: int = find_closest_timestamp_idx(gpx_timestamps, section_endpoint_times[i])
        section_endpoints[i] = idx
        if i > 0:
            idx = find_closest_timestamp_idx(gpx_timestamps,
                                             (section_endpoint_times[i]+section_endpoint_times[i-1]) / 2.0)
            section_midpoints[i-1] = idx

    # Calculate the average over each section. Then calculate the p0 required to adjust
    # the midpoint by this value.
    section_p0: np.ndarray = np.zeros(section_count)
    for i in range(section_count):
        offset: float = np.average(elevation_differences[section_endpoints[i]:section_endpoints[i+1]])
        # offset is the offset to apply at the midpoint.
        p0: float = calculate_p0(gpx_pressures[section_midpoints[i]],
                                 gpx_pressure_elevations[section_midpoints[i]] + offset)
        section_p0[i] = p0

    # Calibrate every elevation based on the section p0 values, interpolating between section points and
    # extrapolating on each end.
    # The same code is used for interpolation and extrapolation.
    # idx is the index into the gpx_timestamps and related arrays.
    # section_idx is the index into the section_midpoints of the current section.
    # section_idx+1 is the index into the section_midpoints of the next section.
    # Interpolation is happening between section_idx and section_idx+1.
    # If we working on a point to the left of section_idx[0] or to the right of section_list[section_count-1],
    # this still works, we are just extrapolating beyond the interpolation range but we still want to
    # use the interpolation range for the slope of the line.
    # We don't use numpy interp() function because it can't handle extrapolation.
    section_idx = 0
    for idx in range(segment_count):
        # Switch to next section if we are there.
        while (section_idx < section_count - 2) \
                    and (idx > section_midpoints[section_idx+1]):
            section_idx += 1

        p0: float = (section_p0[section_idx+1] - section_p0[section_idx]) \
            * (gpx_timestamps[idx] - gpx_timestamps[section_endpoints[section_idx]]) \
            / (gpx_timestamps[section_midpoints[section_idx+1]] - gpx_timestamps[section_endpoints[section_idx]]) \
            + section_p0[section_idx]
        gpx_pressure_elevations[idx] = round(pressure_to_elevation(gpx_pressures[idx], p0), 2)

    return


def calibrate_elevations(gpx_timestamps: np.ndarray, gpx_elevations: np.ndarray,
                         gpx_pressures: np.ndarray, gpx_pressure_elevations: np.ndarray,
                         args: argparse.Namespace) -> None:
    """

    Args:
        gpx_timestamps:     Array of times from the gpx files, converted to datetime.timestamp
        gpx_elevations:     Array of elevations from gpx file
        gpx_pressures:      Array of pressures
        gpx_pressure_elevations: Array of elevations calculated from pressure data. Updated
                                 to contain calibrated elevations.
        args:           Argument list
            calibration_method


    Returns:

    The calibration method can be:
        '0 None',
        'A Average near beginning of file',
        'B Average over entire file',
        'C Linear fit in 1 hour chunks'

    For 0, returns immediately.

    For A and L:
        Calculate the difference between the gpx_elevations and the pressure_elevations.
        1) If the time span of the gpx_times is less than 5 minutes, or there are fewer than 20 points,
           takes the average of the differences and applies that to the pressure_elevations.
        2) Otherwise, if the time span of the gpx_times is less than 10 minutes, takes the average over
           the final 5 minutes or 20 data points, whichever is larger, and applies that to the
           pressure_elevations

    For A:
        If the gpx times are less than 5 minutes, takes the average
    """
    # If calibration method is '0 None', do nothing.
    if args.calibration_method[0].lower() == '0':
        return

    length = len(gpx_timestamps)
    if len(gpx_elevations) != length \
            or len(gpx_pressures) != length \
            or len(gpx_pressure_elevations) != length \
            or length < 2:
        raise IndexError

    # Calculate the elevation differences.
    elevation_differences: np.ndarray = np.subtract(gpx_elevations, gpx_pressure_elevations)

    start_time = gpx_timestamps[0]
    stop_time = gpx_timestamps[length-1]

    # if the gpx file is 0 or 1 points long, do nothing
    if length < 2:
        return

    # if the gpx file is skip_initial_interval or shorter, or less than 20 points, we can't skip
    # the skip_initial_interval. Calibrate to the average of the entire elevation_differences
    if (length < 20) or ((stop_time - start_time) <= skip_initial_interval):
        offset = np.average(elevation_differences)
        add_constant_elevation(gpx_pressure_elevations, gpx_pressures, offset)

    # If the gpx file is shorter than 2 * skip_initial_interval, calibrate to the average of
    # the last skip_interval_interval part of the file.
    elif (stop_time - start_time) <= (2 * skip_initial_interval):
        # side='right' means a[i-1] <= search_time < a[i]
        start_idx = np.searchsorted(gpx_timestamps, stop_time - skip_initial_interval, side='right') - 1
        start_idx = max(start_idx, 0)
        start_idx = min(start_idx, length-1)
        offset = np.average(elevation_differences[start_idx:])
        add_constant_elevation(gpx_pressure_elevations, gpx_pressures, offset)

    # Method A
    # Calibrate to the average of the elevation_differences, but skipping the skip_initial_interval
    # and then averaging for only beginning_average_interval.
    elif args.calibration_method[0].lower() == 'a':
        start_idx = np.searchsorted(gpx_timestamps, start_time + skip_initial_interval, side='right') - 1
        start_idx = max(start_idx, 0)
        start_idx = min(start_idx, length-1)
        end_idx = np.searchsorted(gpx_timestamps, start_time + skip_initial_interval + beginning_average_interval,
                                  side='right')
        offset = np.average(elevation_differences[start_idx:end_idx])
        add_constant_elevation(gpx_pressure_elevations, gpx_pressures, offset)

    # Method B
    # Calibrate to the average of the elevation_differences. But skip the skip_initial_interval.
    # This method is used rather than later methods if the gpx file is shorter than the section
    # time. In this case, the pressure will be essentially constant and attempting to fit a slope
    # to the differences will introduce errors.
    elif (args.calibration_method[0].lower() == 'b') \
            or ((stop_time - start_time) <= (section_interval + skip_initial_interval)):
        start_idx = np.searchsorted(gpx_timestamps, start_time + skip_initial_interval, side='right') - 1
        start_idx = max(start_idx, 0)
        start_idx = min(start_idx, length-1)
        offset = np.average(elevation_differences[start_idx:])
        add_constant_elevation(gpx_pressure_elevations, gpx_pressures, offset)

    # Method C
    # Break the file into sections.
    # At each endpoint, calculate the average elevation_difference over a section_interval surrounding
    # the endpoint. Calibrate the endpoint with that value. Calibrate the points between the endpoints
    # by changing p0 linearly between the end-points.
    elif args.calibration_method[0].lower() == 'c':
        # Pass this off to a separate function.
        calibrate_elevations2(gpx_timestamps, gpx_pressures, gpx_pressure_elevations,
                              elevation_differences)


def convert_segment_points_to_arrays(segment: gpxpy.gpx.GPXTrackSegment) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert the time and elevation data in a segment to numpy ndarrays of float values.

    Removes all points where either the time or elevation is None. This means that the size of the
    returned may be smaller than the size of segment.points, and that the values in the arrays may
    not match up (index to index) with the segment points list.

    Args:
        segment: The segment from the gpx object.

    Returns: 2 element tuple of numpy 1D ndarrays, represent the time and elevation.
    The time is converted to a float timestamp.
    Both arrays have dtype float and are the same size.
    The size of the returned arrays may be smaller than the segment.points list, if there are any
    None values in the data.

    """
    time_list = []
    elevation_list = []
    for point in segment.points:
        if point.time is not None and point.elevation is not None:
            time_list.append(point.time.timestamp())
            elevation_list.append(point.elevation)

    return np.array(time_list), np.array(elevation_list)


def convert_dataframe_to_arrays(sensor_df: pandas.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Optimizations:
    # In my first version, np.searchsorted turned out to be a bottleneck.
    # This version took about 4 seconds to create a pressure list for a gpx file of 844 points
    # with a Tempo Disc dataframe length of 3920 elements. That's shorter than usual -- the
    # typical Tempo Disk dataframe size is 6000 points, so that would take longer.
    # In Version 2 I sliced the date_list and pressure_list to be only those items that include
    # the times in segment_point_times. This reduced the time to 0.3 secs, better than a factor
    # of 10.
    # In version 3 I converted the datetime objects to timestamps, and stored them in a numpy
    # ndarray. I also stored the pressure list in a numpy ndarray. This lets np.searchsorted work
    # on its native type without any conversions. This further reduces the time to 0.022 seconds
    # (more than another factor of 10). If I remove the slicing optimization (version # 2) above,
    # the time only increases slightly to 0.023 seconds, so using ndarrays  is the most important
    # optimization.
    # These timings were performed under the debugger. If you run the application from PyCharm
    # but not using the debugger, the times decrease further to about 0.012 secs for my test case.
    #
    # I now convert both the gpx file and the tempo disk dataframe to numpy ndarrays before
    # substituting pressure-based elevations and calibration.

    # Convert the date from a date to a timestamp.
    # Convert date pressure and temperature lists to ndarrays.
    # All ndarrays are the same length.
    # Remove any entries where either time or pressure is None. There should not be any of these,
    # but just in case.
    # If the temperature value is none, set it to the previous temperature, or 0 if the first entry.
    # Again this shouldn't happen, but check just in case.
    sensor_time_series: pandas.Series = sensor_df['date']
    sensor_pressure_series: pandas.Series = sensor_df['pressure']
    sensor_temperature_series: pandas.Series = sensor_df['temperature']
    sensor_time_list: List[float] = []
    sensor_pressure_list: List[float] = []
    sensor_temperature_list: List[float] = []
    count = min(len(sensor_time_series), len(sensor_pressure_series))
    for i in range(count):
        if sensor_time_series[i] is not None and sensor_pressure_series[i] is not None:
            sensor_time_list.append(sensor_time_series[i].timestamp())
            sensor_pressure_list.append(sensor_pressure_series[i])
            if sensor_temperature_series[i] is not None:
                sensor_temperature_list.append(sensor_temperature_series[i])
            elif i > 0:
                sensor_temperature_list.append(sensor_temperature_series[i-1])
            else:
                sensor_temperature_list.append(0.0)

    return np.array(sensor_time_list), np.array(sensor_pressure_list), np.array(sensor_temperature_list)


gpxtpx_key: str = 'gpxtpx'
gpxtpx_extension: str = 'http://www.garmin.com/xmlschemas/TrackPointExtension/v1'


def add_replace_trackpoint_temperature(point: gpxpy.gpx.GPXTrackPoint, temperature: float):
    """
    Look to see if there is an existing temperature in the track point.
    If so replace it.
    If not, add it.

    Args:
        point:
        temperature:

    Returns:

    """
    # Look to see if there is an existing temperature element
    for ext in point.extensions:
        for ext_child in ext:
            if ext_child.tag[-5:] == 'atemp':
                ext_child.text = str(round(temperature, 1))
                return

    # No existing temp found.
    # point.extensions is a list of lxml.etree._Element objects.
    ext_element = etree.Element('{' + gpxtpx_extension + '}TrackPointExtension')
    sub_element = etree.SubElement(ext_element, '{' + gpxtpx_extension + '}atemp')
    sub_element.text = str(round(temperature, 1))
    point.extensions.append(ext_element)
    return


def replace_elevations_from_pressure(gpx: gpxpy.gpx, sensor_df: pandas.DataFrame, args: argparse.Namespace) -> None:
    """
    Replace the elevations in the gpx file with elevations from a pandas DataFrame.
    The DataFrame comes from a Tempo Disc csv file that has pressure values.

    Args:
        gpx:  Input and output gpx file
        sensor_df:   Input data frame with Tempo Disc data
        args: ArgParse object

    The gpx file is modified in place.

    Returns:
        None
    """
    if not args.merge_pressure and not args.merge_temperature:
        return gpx

    if not gpx or sensor_df is None or sensor_df.empty:
        return gpx

    # For efficiency, convert the pandas DataFrame with the Tempo Disc data to numpy
    # arrays of floats.
    sensor_timestamps, sensor_pressures, sensor_temperatures = convert_dataframe_to_arrays(sensor_df)

    # num_sensor_df_points = len(sensor_df['date'])
    # date_series = sensor_df['date']
    # pressure_series = sensor_df['pressure']

    # default p0
    # 1013.25 hPa is the standard pressure level used in flight at standard flight levels (so all
    # aircraft use the same altimeter setting.
    # It's just a starting point, we will calibrate it.
    p0: float = 1013.25
    # p0 = calculate_p0(893.6, 1143.694)

    for track in gpx.tracks:
        for segment in track.segments:
            # For efficiency, convert gpx file to numpy arrays of floats -- timestamps and elevations.
            # Remove any points which have None for either the time or the elevation.
            gpx_timestamps, gpx_elevations = convert_segment_points_to_arrays(segment)

            # Look up the pressure and temperature for each point.
            # This may remove points if they are outside the barometer data.
            # gpx_timestamps, gpx_elevations, gpx_pressures, and gpx_temperatures will all be the same size (which could be zero).
            # gpx_timestamps and gpx_elevations could be smaller than before.
            gpx_timestamps, gpx_elevations, gpx_pressures, gpx_temperatures, = \
                get_point_data(gpx_timestamps, gpx_elevations,
                               sensor_timestamps, sensor_pressures, sensor_temperatures)

            if args.merge_pressure:
                # Calculate new (uncalibrated) elevations from the pressure data.
                gpx_pressure_elevations: np.ndarray = np.zeros_like(gpx_pressures)
                gpx_pressures_to_elevations(gpx_pressure_elevations, gpx_pressures, p0)

                # Perform calibration if requested
                calibrate_elevations(gpx_timestamps, gpx_elevations, gpx_pressures, gpx_pressure_elevations, args)

            if args.merge_temperature:
                # Make sure there is a TrackPointExtension enabled, so we can insert temperatures if we have any.
                if (gpx_temperatures is not None) \
                        and (len(gpx_temperatures) > 0) \
                        and (gpxtpx_key not in gpx.nsmap):
                    gpx.nsmap[gpxtpx_key] = gpxtpx_extension

            # Store the results back into gpx
            if (args.merge_pressure or args.merge_temperature) \
                    and (len(gpx_timestamps) > 0):
                pressure_idx = 0
                for idx, point in enumerate(segment.points):
                    if point.time.timestamp() >= gpx_timestamps[pressure_idx]:
                        if args.merge_pressure and pressure_idx < len(gpx_pressure_elevations):
                            point.elevation = gpx_pressure_elevations[pressure_idx]
                        if args.merge_temperature \
                                and gpx_temperatures is not None \
                                and pressure_idx < len(gpx_temperatures):
                            add_replace_trackpoint_temperature(point, gpx_temperatures[pressure_idx])
                        pressure_idx += 1

# I have broken elevation from sensor pressure.
# Need to fix this.
