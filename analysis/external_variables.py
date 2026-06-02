# Tools for reading external variables

import os
import sys
import pickle
import numpy as np
import awkward as ak


def find_external_files(input_files, external_variable_dir, verbose=False):
    # find external variable files corresponding to input files
    external_dict = {}
    for input_file in input_files:
        tag = input_file.replace('/', '').replace('.root', '')
        external_variable_file = os.path.join(external_variable_dir, tag+'.pkl')
        if os.path.exists(external_variable_file): external_dict[input_file] = external_variable_file
        else:
            external_dict[input_file] = None
            if verbose:
                msg = f'WARNING: expected external variable file {external_variable_file} does not exist, returning None.'
                print(msg)
    return external_dict


def read_external_variables(input_files, external_variable_dir):
    # read external variables
    # note: preliminary implementation, to make more robust
    temp = []
    variable_names = None
    external_file_dict = find_external_files(input_files, external_variable_dir)
    for input_file, external_file in external_file_dict.items():
        if external_file is None or not os.path.exists(external_file):
            raise Exception(f'Expected external variable file {external_file} does not exist.')
        with open(external_file, 'rb') as f:
            content = pickle.load(f)
        temp.append(content)
        # check if variable names are consistent
        candidate_variable_names = list(content.keys())
        if variable_names is None: variable_names = candidate_variable_names
        else:
            if variable_names != candidate_variable_names:
                msg = 'Inconsistent variables found:'
                msg += ' {candidate_variable_names} vs {variable_names}.'
                raise Exception(msg)
    # concatenate results from all files
    external_variables = {}
    for key in variable_names:
        values = [el[key] for el in temp]
        if any(isinstance(value, ak.Array) for value in values):
            external_variables[key] = ak.concatenate(values)
        else:
            external_variables[key] = np.concatenate(values)
    # return the result
    return external_variables
