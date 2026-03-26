# Script to quickly run a number of plotting commands

import os
import sys
import json
import argparse

thisdir = os.path.abspath(os.path.dirname(__file__))
topdir = os.path.abspath(os.path.join(thisdir, '../'))
sys.path.append(topdir)

import tools.condortools as ct
import tools.slurmtools as st


if __name__=='__main__':

    # read command line args
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--samples', required=True)
    parser.add_argument('-c', '--config', required=True, nargs='+')
    parser.add_argument('-k', '--keys', default=None, nargs='+')
    parser.add_argument('-r', '--runmode', default='local', choices=['local', 'condor'])
    args = parser.parse_args()

    # read samples
    with open(args.samples, 'r') as f:
        samples = json.load(f)
    print(f'Read sample list: {args.samples}.')
    print(f'Found following samples:')
    print(json.dumps(samples, indent=2))

    # read config(s)
    config = {}
    for configfile in args.config:
        with open(configfile, 'r') as f:
            this_config = json.load(f)
            config.update(this_config)
    print(f'Read config file(s): {args.config}.')
    print(f'Found following config:')
    print(json.dumps(config, indent=2))

    # filtering
    if args.keys is not None:
        config = {key: val for key, val in config.items() if key in args.keys}
        print(f'Found following config (after filtering with keys {args.keys}):')
        print(json.dumps(config, indent=2))

    # make base command
    base_args = {
        'sim': samples.get('sim', None),
        'data': samples.get('data', None),
        'variables': ['variables/variables_event.json'],
        'outputdir': 'output_test2',
        'objectselection': ['selections/selection_Rb.json'],
        'eventselection': 'selections/selection.json',
        'year': '1994',
        'xsections': 'cross-sections/cross_sections.json',
        'merge': 'merging/merging.json',
        'split': 'merging/splitting.json',
        'dolog': True
    }

    # make commands based on config
    cmds = []
    for key, settings in config.items():
        this_args = base_args.copy()
        this_args.update(settings)
        cmd = 'python plot.py'
        # loop over arguments
        for arg, val in this_args.items():
            # parse argument
            if val is None: continue
            elif isinstance(val, bool) and val: cmd += f' --{arg}'
            elif isinstance(val, str): cmd += f' --{arg} {val}'
            elif isinstance(val, list): cmd += f' --{arg} {" ".join(val)}'
            else: raise Exception(f'Value of argument {arg} not recognized: {val} ({type(val)})')
            # check argument
            if arg in ['variables', 'objectselection', 'eventselection', 'xsections', 'merge', 'split']:
                if not isinstance(val, list): val = [val]
                for el in val:
                    if not os.path.exists(el):
                        raise Exception(f'{el} does not exist.')
        cmds.append(cmd)

    # run or submit commands
    if args.runmode=='local':
        for cmd in cmds:
            print(cmd)
            os.system(cmd)
    elif args.runmode=='condor':
        env_script = os.path.abspath('../../../setup.sh')
        env_cmd = f'source {env_script}'
        for cmd in cmds:
            ct.submitCommandAsCondorJob('cjob_analysis', cmd,
              jobflavour='workday', conda_activate=env_cmd)
