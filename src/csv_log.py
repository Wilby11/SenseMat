'''CSV helper functions'''
import json


def generate_header(config, serial_identifier):
    '''Generate header for CSV file recording.'''
    header = "RECV_TIME"
    header += ",TRIG_CNT"
    header += ",B_TIME"
    mat_config = next(mat for mat in config if mat['id'] == serial_identifier)
    # for i, mat_config in enumerate(config):
    for col in range(0, mat_config['rx']):
        for row in range(0, mat_config['tx']):
            header += f",S_{col}_{row}"

    header += ",S_mean"
    header += ",TTL_time"
    header += ",TTL_state"
    header += '\n'
    return header


def generate_configuration_comment(config):
    '''Include configuration in data recording.'''
    return f"# config = {json.dumps(config)}\n"


def parse_configuration_comment(comment_line):
    '''Read config comment from csv file.'''
    config_comment_start = '# config = '
    if comment_line.startswith(config_comment_start):
        # Strips off the newline
        config_value = comment_line[len(config_comment_start):-1]
        return json.loads(config_value)
    return None
