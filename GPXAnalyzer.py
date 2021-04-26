# Copyright 2021 Don McClimans
# This program is licensed under the MIT license. See the License.txt file for details

# If running in interactive mode, display a splash screen.
# Do this before most imports so it displays while you are importing.
# This helps when you run pyinstaller to create a windows application folder. The resulting
# exe can take 10 or more seconds before the main window displays. This splash screen
# displays within 1 to 2 seconds, and gives some feedback to the user that they have
# started the application, and need to wait.
import sys
# noinspection PyPep8Naming
import PySimpleGUI as sg

version_string = 'Version 1.0'
if len(sys.argv) <= 1:
    # PySimpleGUI (and tkinter) only supports  png and gif.
    splash_layout = [
                [sg.Image('splashscreen.png')],
                [sg.Text('GPXAnalyzer', size=(75, 1)), sg.Text(version_string)]
                ]
    splash_window = sg.Window('GPXAnalyzer', splash_layout, no_titlebar=True, keep_on_top=True)
    splash_window.read(timeout=10)
    # Sometimes the window displays a completely black image instead of the desired image.
    # Refresh seems to fix that.
    splash_window.refresh()

# noinspection PyUnresolvedReferences
from typing import List, Set, Dict, Tuple, Optional
import argparse
import gpxpy
import GPXAnalyze
import configparser
import appdirs
import os
import traceback
import sys
import copy
import pandas


def show_arguments(args: argparse.Namespace) -> None:
    print(f'Units: {args.units}')
    if (args.merge_temperature or args.merge_pressure) and args.sensor_file:
        print(f'Sensor file: {args.sensor_file}')
        if args.merge_pressure:
            print(f'Substitute elevations from sensor file:')
            print(f'Sensor calibration method: {args.calibration_method}')
        if args.merge_temperature:
            print('Merge temperatures from sensor file to output gpx file')
    filter_count_str: str = ''
    if args.elevation_filter_count > 1 and args.elevation_filter[0:1].lower() != '0':
        filter_count_str = f' repeated {args.elevation_filter_count} times'
    print(f'Elevation filter: {args.elevation_filter}{filter_count_str}')
    print(f'Difference file: {args.difference_file}')
    if args.output_suffix and args.output_suffix.strip():
        print(f'Output filename suffix: {args.output_suffix}')
    else:
        print('No output file')
    if args.plot_input or args.plot_before_difference or args.plot_difference_file or args.plot_output:
        print(f'Show plot: {args.show_plot}')
        if args.plot_suffix and args.plot_suffix.strip():
            print(f'Save plot with filename suffix: {args.plot_suffix}')
        else:
            print('Do not automatically save plot')
    else:
        print('No elevation plot')
    if args.plot_input:
        print(f'Plot input elevations')
    if args.plot_before_difference:
        print(f'Plot before difference  elevations')
    if args.plot_difference_file:
        print(f'Plot difference file elevations')
    if args.plot_output:
        print(f'Plot output elevations')
    # if args.gpxpy_up_down:
    #     print(f'Calculate gpxpy.get_uphill_downhill()')
    if args.ceg_threshold >= 0:
        print(f'Calculate CEG/CEL. Threshold {args.ceg_threshold}. All values: {args.all_ceg}.'
              f' Plot CEG: {args.plot_ceg}.')
    if args.plot_ceg and args.all_ceg and args.ceg_threshold >= 2:
        print('Plot CEG values')


def process_files(args: argparse.Namespace, is_interactive: bool):
    """Process a list of filenames"""

    show_arguments(args)

    filenames: list[str] = args.files

    if not filenames or len(filenames) == 0:
        print('No filenames specified')
        return

    # Read the difference file if one is specified
    difference_filename: str = args.difference_file
    difference_gpx: gpxpy.gpx = None
    if difference_filename:
        try:
            with open(difference_filename, 'r', encoding='utf-8-sig') as gpx_file:
                difference_gpx = gpxpy.parse(gpx_file)
        except (IOError, gpxpy.gpx.GPXException, gpxpy.gpx.GPXXMLSyntaxException, ValueError, Exception) as exc:
            print(f'Cannot read difference file:\n    {difference_filename}\nError: {str(exc)}')
            difference_gpx = None

    # Read the sensor file if one is specified
    sensor_df: Optional[pandas.DataFrame] = \
        GPXAnalyze.read_csv_pressure_file(args.sensor_file, is_interactive)

    for input_filename in filenames:
        if not (input_filename and input_filename.strip()):
            print('No filename specified')
            return

        # noinspection PyBroadException
        # gpxpy.geo.py and gpxpy.gpxfield.py raise various exceptions, including plain exceptions,
        try:
            with open(input_filename, 'r', encoding='utf-8-sig') as gpx_file:
                input_gpx = gpxpy.parse(gpx_file)
        except (IOError, gpxpy.gpx.GPXException, gpxpy.gpx.GPXXMLSyntaxException, ValueError, Exception) as exc:
            print(f'Cannot read file:\n    {input_filename}\nError: {str(exc)}')
            return

        print(f'\nFile: {input_filename}')

        output_gpx: Optional[gpxpy.gpx] = copy.deepcopy(input_gpx)

        GPXAnalyze.replace_elevations_from_pressure(output_gpx, sensor_df, args)

        for i in range(0, args.elevation_filter_count):
            GPXAnalyze.filter_elevation(output_gpx, args.elevation_filter)

        pre_difference_gpx: gpxpy.gpx = copy.deepcopy(output_gpx)

        GPXAnalyze.subtract_difference(output_gpx, difference_gpx, difference_filename)

        GPXAnalyze.write_gpx_file(output_gpx, input_filename, args.output_suffix)

        GPXAnalyze.plot_elevations(input_gpx, input_filename, pre_difference_gpx, difference_gpx,
                                   output_gpx, args, is_interactive)

        GPXAnalyze.print_stats(output_gpx, args, is_interactive)


def load_config(config_filename) -> configparser:
    """Load settings from config file.

    The config file is an ini file with a single section [Settings]
    Only used when GUI displayed
    The gui version will preserve settings from one execution to the next.
    """

    config: configparser = configparser.ConfigParser()
    config.read(config_filename)
    if not config.has_section('Settings'):
        config.add_section('Settings')

    return config


def save_config(config, values: Dict, config_filename):
    """Save config to config_filename.

    Will create directory if necessary.
    Only used when GUI displayed
    """
    # Save values in config file
    # If you use the Exit button, the values dictionary will be set to the values from the form.
    # If you use the X button in upper right corner of window, you can get various behavior
    # 1. Correct valuers
    # 2. All values in the dictionary are set to None
    # 3. Values itself is set to None (rather than a dictionary).
    # You may get (2) or (3) even if you have made changes to the settings (bad).
    if values is None or values['units'] is None:
        return

    settings = config['Settings']
    settings['files'] = '' if values['files'] is None else values['files']
    settings['units'] = values['units']
    settings['sensor_file'] = values['sensor_file']
    settings['merge_pressure'] = 'True' if values['merge_pressure'] else 'False'
    settings['merge_temperature'] = 'True' if values['merge_temperature'] else 'False'
    settings['calibration_method'] = values['calibration_method']
    settings['elevation_filter'] = str(values['elevation_filter'])
    settings['elevation_filter_count'] = str(values['elevation_filter_count'])
    settings['difference_file'] = '' if values['difference_file'] is None else values['difference_file']
    settings['output_suffix'] = values['output_suffix']
    settings['show_plot'] = 'True' if values['show_plot'] else 'False'
    settings['plot_suffix'] = values['plot_suffix']
    settings['plot_input'] = 'True' if values['plot_input'] else 'False'
    settings['plot_before_difference'] = 'True' if values['plot_before_difference'] else 'False'
    settings['plot_difference_file'] = 'True' if values['plot_difference_file'] else 'False'
    settings['plot_output'] = 'True' if values['plot_output'] else 'False'
    # settings['gpxpy_up_down'] = 'True' if values['gpxpy_up_down'] else 'False'
    settings['ceg'] = 'True' if values['ceg'] else 'False'
    settings['ceg_threshold'] = values['ceg_threshold']
    settings['all_ceg'] = 'True' if values['all_ceg'] else 'False'
    settings['plot_ceg'] = 'True' if values['plot_ceg'] else 'False'

    directory = os.path.dirname(config_filename)
    if not os.path.exists(directory):
        os.makedirs(directory)
    with open(config_filename, 'w') as configfile:
        config.write(configfile)


def convert_settings_to_args(values: dict) -> argparse.Namespace:
    """Convert GUI settings to command line args"""
    args = argparse.Namespace()
    args.files = values['files'].split(';')
    args.units = values['units'].lower()
    args.sensor_file = values['sensor_file']
    args.merge_pressure = values['merge_pressure']
    args.merge_temperature = values['merge_temperature']
    args.calibration_method = values['calibration_method']
    args.elevation_filter = values['elevation_filter']
    args.elevation_filter_count = int(values['elevation_filter_count'])
    args.output_suffix = values['output_suffix']
    args.difference_file = values['difference_file']
    args.show_plot = values['show_plot']
    args.plot_suffix = values['plot_suffix']
    args.plot_input = values['plot_input']
    args.plot_before_difference = values['plot_before_difference']
    args.plot_difference_file = values['plot_difference_file']
    args.plot_output = values['plot_output']
    # args.gpxpy_up_down = values['gpxpy_up_down']
    args.ceg_threshold = float(values['ceg_threshold'])
    if not values['ceg']:
        args.ceg_threshold = -1
    args.all_ceg = values['all_ceg']
    args.plot_ceg = values['plot_ceg']
    return args


def gui():
    """Display GUI"""
    # sg.preview_all_look_and_feel_themes()

    # The GUI redirects stdout to a control in the GUI. That disappears on
    # exit, so if there is an exception we never see the information.
    # To handle this, catch all exceptions and display a popup.

    # Restore settings from settings file.
    config_filename: str =\
        os.path.join(appdirs.user_config_dir('GPXAnalyzer', '', None, True), 'GPXAnalyzer.ini')

    config: configparser.ConfigParser = load_config(config_filename)
    settings: configparser.SectionProxy = config['Settings']
    default_files = settings.get('files', '')
    default_units: str = settings.get('units', 'Metric')
    default_sensor_file: str = settings.get('sensor_file')
    default_merge_pressure: bool = settings.getboolean('merge_pressure', fallback=False)
    default_merge_temperature: bool = settings.getboolean('merge_temperature', fallback=False)
    default_calibration_method: str = settings.get('calibration_method', 'C Linear fit in 1 hour sections')
    default_filter: str = settings.get('elevation_filter', '0 None')
    default_filter_count: str = settings.get('elevation_filter_count', '1')
    default_difference_file: str = settings.get('difference_file', '')
    default_output_suffix: str = settings.get('output_suffix')
    default_show_plot: bool = settings.getboolean('show_plot', fallback=False)
    default_plot_suffix: str = settings.get('plot_suffix')
    default_plot_input: bool = settings.getboolean('plot_input', fallback=False)
    default_plot_before_difference: bool = settings.getboolean('plot_before_difference', fallback=False)
    default_plot_difference_file: bool = settings.getboolean('plot_difference_file', fallback=False)
    default_plot_output: bool = settings.getboolean('plot_output', fallback=False)
    # default_gpxpy_up_down: bool = settings.getboolean('gpxpy_up_down', fallback=False)
    default_ceg: bool = settings.getboolean('ceg', fallback=False)
    default_ceg_threshold: str = str(GPXAnalyze.user_units_to_meters(settings.getfloat('ceg_threshold', fallback=2.0),
                                                                     default_units))
    default_all_ceg: bool = settings.getboolean('all_ceg', fallback=False)
    default_plot_ceg: bool = settings.getboolean('plot_ceg', fallback=False)

    sg.theme('Default1')

    layout = [[sg.Text('Input:'),
               sg.Input(size=(85, 1), key='files', default_text=default_files),
               sg.FilesBrowse()],
              [sg.Text('Units: ', size=(10, 1)),
               sg.Combo(values=['English', 'Metric'], default_value=default_units, key='units',
                        readonly=True, size=(20, 0))],
              [sg.Text('Sensor file:'),
               sg.Input(size=(76, 1), key='sensor_file', default_text=default_sensor_file),
               sg.FileBrowse()],
              [sg.Checkbox('Calculate elevation from sensor pressure', key='merge_pressure',
                           default=default_merge_pressure),
               sg.Text('Calibration method:'),
               sg.Combo(values=[
                   '0 None',
                   'A Average over minutes 5 to 10',
                   'B Average over entire file except first 5 minutes',
                   'C Linear fit in 1 hour sections'
               ], default_value=default_calibration_method, key='calibration_method', readonly=True, size=(40, 1))],
              [sg.Checkbox('Merge sensor temperature', key='merge_temperature', default=default_merge_temperature)],
              [sg.Text('Elevation filter:', size=(10, 1)),
               sg.Combo(values=[
                   '0 None',
                   'A 3-point average',
                   'B 5-point average',
                   'C 7-point average',
                   'D (0.3, 0.4, 0.3) weighted average',
                   # 'E (0.4, 0.2, 0.4) weighted average',
                   # 'F gpxpy.smooth'
               ], default_value=default_filter, key='elevation_filter', readonly=True, size=(40, 1)),
               sg.Text('Run filter '),
               sg.Spin([i for i in range(1, 11)], initial_value=default_filter_count, size=(3, 1),
                       key='elevation_filter_count'),
               sg.Text(' times')],
              [sg.Text('Difference file:', size=(10, 1)),
               sg.Input(size=(78, 1), key='difference_file', default_text=default_difference_file),
               sg.FileBrowse()],
              [sg.Text('Output file and track name suffix: ', size=(25, 1)),
               sg.Input(size=(20, 1), default_text=default_output_suffix, key='output_suffix')],
              [sg.Checkbox('Show plot', default=default_show_plot, key='show_plot'),
               sg.Text('Plot file suffix.ext: '),
               sg.Input(size=(20, 1), default_text=default_plot_suffix, key='plot_suffix')],
              [sg.Text('Plot: '),
               sg.Checkbox('Input', default=default_plot_input, key='plot_input'),
               sg.Checkbox('Before difference', default=default_plot_before_difference,
                           key='plot_before_difference'),
               sg.Checkbox('Difference file', default=default_plot_difference_file, key='plot_difference_file'),
               sg.Checkbox('Output', default=default_plot_output, key='plot_output')],
              # [sg.Checkbox('Calculate gpxpy.get_uphill_downhill()', default=default_gpxpy_up_down,
              #              key='gpxpy_up_down')],
              [sg.Checkbox('Calculate CEG/CEL. Threshold: ', default=default_ceg, key='ceg'),
               sg.Input(size=(10, 1), default_text=default_ceg_threshold, key='ceg_threshold'),
               sg.Checkbox('All values up to threshold', default=default_all_ceg, key='all_ceg'),
               sg.Checkbox('Plot', default=default_plot_ceg, key='plot_ceg')],
              [sg.Button('Process')],
              [sg.Output(size=(100, 30), key='output')],
              [sg.Button('Exit')]]

    window_title = 'GPX Analyzer (' + version_string + ')'
    window = sg.Window(window_title, layout)

    while True:  # Event Loop
        event, values = window.read()
        # print(event, values)
        if event == sg.WIN_CLOSED or event == 'Exit':
            save_config(config, values, config_filename)
            break

        if event == 'Process':
            window['output'].Update('')
            save_config(config, values, config_filename)
            args = convert_settings_to_args(values)
            process_files(args, True)

    window.close()


def parse_command_line() -> argparse.Namespace:
    """Define Parse command line parameters"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--units', type=str.lower, choices=['english', 'metric'],
                        default='metric', help='Specify units')
    parser.add_argument('--sensor_file', default='',
                        help='Substitute elevations from Tempo disc file')
    parser.add_argument('--merge_temperature', default=False, action='store_true',
                        help='Merge temperature from sensor file into gpx file')
    parser.add_argument('--merge_pressure', default=False, action='store_true',
                        help='Merge pressure from sensor file and calculate new elevations')
    parser.add_argument('--calibration_method', choices=[
        '0 None',
        'A Average over minutes 5 to 10',
        'B Average over entire file except first 5 minutes',
        'C Linear fit in 1 hour sections'],
        default='C Linear fit in 1 hour sections', help='Calibration method for tempo disc files')
    parser.add_argument('--elevation_filter', type=str.lower, choices=[
        '0 None',
        'A 3-point average',
        'B 5-point average'
        'C 7-point average'
        'D (0.3, 0.4, 0.3) weighted average',
        # 'E (0.4, 0.2, 0.4) weighted average',
        # 'F gpxpy.smooth'
    ],
        default='0 None', help='Filter to apply to elevation values')
    parser.add_argument('--elevation_filter_count', type=int, default=1,
                        help='Number of times to run filter')
    parser.add_argument('--difference_file', default='',
                        help='File to subtract from each input file')
    parser.add_argument('--output_suffix', default='',
                        help='Suffix to add to end of filename when writing output gpx file. '
                             'If not specified, no output')
    parser.add_argument('--show_plot', default=False, action='store_true',
                        help='Show interactive plot')
    parser.add_argument('--plot_suffix', default='',
                        help='Suffix with extension to add to end of filename to write plot file. '
                             'If not specified, no plot file')
    parser.add_argument('--plot_input', default=False, action='store_true',
                        help='Plot input elevation')
    parser.add_argument('--plot_before_difference', default=False, action='store_true',
                        help='Plot before difference elevation')
    parser.add_argument('--plot_difference_file', default=False, action='store_true',
                        help='Plot difference file elevation')
    parser.add_argument('--plot_output', default=False, action='store_true',
                        help='Plot output elevation (after filter and difference)')
    # parser.add_argument('--gpxpy_up_down', default=False, action='store_true',
    #                     help='Calculate gpxpy.get_uphill_downhill')
    parser.add_argument('--ceg_threshold', type=int, default=2,
                        help='Threshold distance to ignore in CEG/CEL (in user units)')
    parser.add_argument('--all_ceg', default=False, action='store_true',
                        help='Calculate CEG/CEL for all values up to threshold')
    parser.add_argument('--plot_ceg', default=False, action='store_true',
                        help='Plot all CEG values up to threshold')
    parser.add_argument('files', help='', nargs='*')
    args = parser.parse_args()
    return args


def main():
    """Parse command line parameters or display GUI to get parameters"""
    args = parse_command_line()

    if splash_window is not None:
        splash_window.close()

    if len(args.files) == 0:
        # Display GUI
        try:
            gui()
        except: # noqa
            # Because stdout is redirected to the output control in the PiSimpleGUI interface,
            # and the GUI disappears when things fail, we catch all exceptions and display them
            # in a popup.
            error = sys.exc_info()
            sg.PopupOK('Unexpected error:\n  '
                       + error[1].__class__.__name__ + ':  '
                       + str(error[1]) + '\n\n'
                       + traceback.format_exc())
    else:
        # Run as command line app.
        process_files(args, False)


# Run the main program when this file is executed as a script.
if __name__ == '__main__':
    main()
