import os
import sys
import glob
import json
import uproot
import awkward as ak


def find_files(samplelist, verbose=True):
    '''
    Read a sample list and find all files.
    Input arguments:
    - samplelist: can be either
        - path to a json file holding a dict of the form
          {tag: [list of files], ...}
          (where the elements in the file list may contain unix-style wildcards)
        - path to a json file holding a list of files
          (where the elements in the list may contain unix-style wildcards)
        - path to a directory (all root files the directory will be read)
        - path to a root file
        - list of any of the above
    Returns:
    - a dict of the form {tag: [list of files], ...}
    '''
    sampledict = {}
    # allow for either one or multiple sample lists provided
    samplelists = samplelist
    if isinstance(samplelist, str): samplelists = [samplelist]
    # loop over sample lists
    for samplelist in samplelists:
        # case in which it is a directory
        if os.path.isdir(samplelist):
            rootfiles = sorted([os.path.join(samplelist, f) for f in os.listdir(samplelist)
                         if f.endswith('.root')])
            if len(rootfiles)==0:
                msg = 'WARNING in tools.samplelisttools.find_files:'
                msg += f' path {samplelist} did not match any files.'
                print(msg)
            for f in rootfiles:
                key = os.path.basename(f).replace('.root', '')
                if key in sampledict.keys(): sampledict[key].append(f)
                else: sampledict[key] = [f]
        # case where it is a root file
        elif samplelist.endswith('.root'):
            rootfile = samplelist
            key = os.path.basename(rootfile).replace('.root', '')
            if key in sampledict.keys(): sampledict[key].append(rootfile)
            else: sampledict[key] = [rootfile]
        # case in which it is a json file 
        else:
            # read sample list
            with open(samplelist, 'r') as f:
                samples = json.load(f)
            # handle case where samples is a list
            if isinstance(samples, list):
                # loop over paths
                for path in samples:
                    key = os.path.basename(path).replace('.root', '')
                    files = sorted(glob.glob(path))
                    if len(files)==0:
                        msg = 'WARNING in tools.samplelisttools.find_files:'
                        msg += f' path {path} did not match any files.'
                        print(msg)
                    if key in sampledict.keys(): sampledict[key] += files
                    else: sampledict[key] = files
            # handle case where samples is a dict
            elif isinstance(samples, dict):
                # loop over tags and paths
                for tag, paths in samples.items():
                    if tag in sampledict.keys():
                        msg = 'WARNING in tools.samplelisttools.find_files:'
                        msg += f' found multiple instances of the same sample tag {tag}.'
                        print(msg)
                    else: sampledict[tag] = []
                    for path in paths:
                        files = sorted(glob.glob(path))
                        if len(files)==0: 
                            msg = 'WARNING in tools.samplelisttools.find_files:'
                            msg += f' path {path} did not match any files.'
                            print(msg)
                        sampledict[tag] += files
    if verbose:
        print(f'Found following samples in {samplelist}:')
        for key, val in sampledict.items(): print(f'  - {key} ({len(val)} files)')
    return sampledict


def read_sampledict(sampledict, treename=None, branches=None,
        entry_start=None, entry_stop=None, verbose=True):
    
    # initializations
    event_dict = {}
    if entry_start is not None:
        entry_start = int(entry_start)
        if entry_start < 0: entry_start = None
    if entry_stop is not None:
        entry_stop = int(entry_stop)
        if entry_stop < 0: entry_stop = None
    
    # loop over samples
    for tag, files in sampledict.items():
        events = []
        # loop over files
        for filename in files:
            # open file and read requested branches
            if treename is not None: filename += f':{treename}'
            f = uproot.open(filename)
            valid_branches = None
            if branches is not None:
                valid_branches = branches[:]
                for branch in branches:
                    if not branch in f.keys():
                        valid_branches.remove(branch)
                        msg = 'WARNING in tools.samplelisttools.read_sampledict:'
                        msg += f' branch {branch} not found in {filename};'
                        msg += ' will skip reading this branch.'
                        print(msg)
            #part = f.arrays(valid_branches, entry_start=entry_start, entry_stop=entry_stop,
            #        library='ak')
            print("FILE:", f.file.file_path if hasattr(f, "file") else f)
            print("ENTRIES:", f.num_entries)
            print("valid_branches:", valid_branches)

            bad_branches = []
            for br in valid_branches:
                try:
                    part = f.arrays(
                        valid_branches,
                        entry_start=entry_start,
                        entry_stop=entry_stop,
                        library="ak"
                    )
                except Exception as e:
                    print("Skipping bad file:", f.file.file_path if hasattr(f, "file") else f)
                    print(repr(e))
                    continue

            #if bad_branches:
            #    raise RuntimeError(
            #        f"Bad branches in file {f.file.file_path if hasattr(f, 'file') else f}: {bad_branches}"
            #    )

            #part = f.arrays(
            #    valid_branches,
            #    entry_start=entry_start,
            #    entry_stop=entry_stop,
            #    library="ak"
            #)
            events.append(part)
        
        # merge events for all files per sample
        events = ak.concatenate(events)
        event_dict[tag] = events

    # printouts for logging
    if verbose:
        print('Found following samples:')
        for key in event_dict.keys():
            nfiles = len(sampledict[key])
            nevents = len(event_dict[key])
            nbranches = len(event_dict[key].fields)
            print(f'  - {key} ({nfiles} files, {nevents} events, {nbranches} branches)')
    return event_dict


def read_num_entries(sampledict, treename=None, verbose=True):

    # initializations
    num_entries_dict = {}

    # loop over samples
    for tag, files in sampledict.items():
        num_entries = {}

        # loop over files
        for filename in files:
            # open file and find number of entries
            origfilename = filename
            if treename is not None: filename += f':{treename}'
            f = uproot.open(filename)
            num_entries[origfilename] = f.num_entries

        # add result to dict
        num_entries_dict[tag] = num_entries

    # printouts for logging
    if verbose:
        print('Found following number of entries:')
        for key in num_entries_dict.keys():
            nfiles = len(num_entries_dict[key].keys())
            nentries = sum(num_entries_dict[key].values())
            print(f'  - {key} ({nfiles} files, {nentries} entries)')
    return num_entries_dict


def read_branchnames(sampledict, treename=None, verbose=True):

    branchnames = None
    bset = None
    # loop over samples
    for tag, files in sampledict.items():
        # loop over files
        for filename in files:
            # open file and find branches
            if treename is not None: filename += f':{treename}'
            f = uproot.open(filename)
            thisbranchnames = f.keys()
            if branchnames is None:
                branchnames = thisbranchnames
                bset = set(branchnames)
            else:
                thisbset = set(thisbranchnames)
                if thisbset != bset:
                    print('WARNING: found inconsistent branch names.')

    # printouts for logging
    if verbose:
        print('Found following branches:')
        print(branchnames)
    return branchnames


def read_samplelist(samplelist, treename=None, branches=None,
        entry_start=None, entry_stop=None, verbose=True):
    file_dict = find_files(samplelist, verbose=False)
    event_dict = read_sampledict(file_dict, treename=treename, branches=branches,
            entry_start=entry_start, entry_stop=entry_stop, verbose=False)
    if verbose:
        print(f'Found following samples in {samplelist}:')
        for key in file_dict.keys():
            nfiles = len(file_dict[key])
            nevents = len(event_dict[key])
            nbranches = len(event_dict[key].fields)
            print(f'  - {key} ({nfiles} files, {nevents} events, {nbranches} branches)')
    return event_dict
