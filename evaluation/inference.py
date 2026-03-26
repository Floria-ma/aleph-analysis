import os
import sys
import json
import pickle
import argparse
import numpy as np
import awkward as ak
import uproot
import matplotlib.pyplot as plt

thisdir = os.path.abspath(os.path.dirname(__file__))
topdir = os.path.abspath(os.path.join(thisdir, '../'))
sys.path.append(topdir)

from tools.samplelisttools import find_files
from tools.samplelisttools import read_sampledict
from tools.samplelisttools import read_samplelist
from tools.samplelisttools import read_num_entries
from tools.plottools import merge_sampledict
from evaluation.inferencetools import infer_events, add_variables
from analysis.objectselection import load_objectselection
from analysis.objectselection import apply_objectselection
from analysis.eventselection import load_eventselection
from analysis.eventselection import get_variable_names
from analysis.eventselection import get_selection_mask


if __name__=='__main__':

    # read command line args
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--samples', required=True, nargs='+')
    parser.add_argument('-m', '--model', required=True)
    parser.add_argument('-p', '--preprocess', required=True)
    parser.add_argument('-o', '--outputdir', default=None)
    parser.add_argument('-t', '--treename', default=None)
    parser.add_argument('--translation', default=None)
    parser.add_argument('--objectselection', default=None)
    args = parser.parse_args()

    # load the preprocess dict
    with open(args.preprocess, 'r') as f:
        prepdict = json.load(f)

    # load the object selection dict and parse selection
    objectselection = None
    if args.objectselection is not None:
        objectselection = load_objectselection(args.objectselection)

    # read translation dict
    translation = None
    if args.translation is not None:
        with open(args.translation, 'r') as f:
            translation = json.load(f)

    # define variables to read
    branches_to_read = []
    # add input variables defined in the preprocess dict
    for key in prepdict['input_names']:
        for varname in prepdict[key]['var_names']:
            if translation is not None: varname = translation.get(varname, varname)
            branches_to_read.append(varname)

        extra_branches = [
        "SecondaryVertices_mass",
        "Event_njets",
        "Jets_nChargedHad",
        "Jets_px", 
        "Jets_py",
        "Jets_pz",
        "genEventType",
        "SecondaryVertices_nTracks",
        "SecondaryVertices_ndof",
        "SecondaryVertices_dxy",
        "SecondaryVertices_dz",
        "SecondaryVertices_xrel",
        "SecondaryVertices_yrel",
        "SecondaryVertices_zrel",
        "SecondaryVertices_prel",
        "SecondaryVertices_correctedMass",
        "V0Candidates_mass",
        "V0Candidates_nTracks",
        "V0Candidates_ndof",
        "V0Candidates_xrel",
        "V0Candidates_yrel",
        "V0Candidates_zrel",
        "V0Candidates_prel",
    ]
    branches_to_read += extra_branches
    branches_to_read = list(set(branches_to_read))

    # add variables needed for runtime variable definitions
    newvarnames = add_variables(None, names_only=True)
    for varname in newvarnames['input_names']:
        branches_to_read.append(varname)
    # add branches needed for selection
    if objectselection is not None:
        branches_to_read += get_variable_names(objectselection[0])
    # add at least one per-jet variable (needed to get the correct shape during inference)
    branches_to_read.append('Jets_pt')
    # remove potential duplicates
    branches_to_read = list(set(branches_to_read))
    # filter out runtime variables
    for varname in newvarnames['output_names']:
        if varname in branches_to_read: branches_to_read.remove(varname)
    print('Found following branches to read:')
    print(sorted(branches_to_read))

    # find samples
    print('Finding samples...')
    sampledict = find_files(args.samples)

    # merge into one
    mergedict = {'_': '*'}
    sampledict = merge_sampledict(sampledict, mergedict)
   
    # read all events 
    for _, files in sampledict.items():
        for fidx, file in enumerate(files):
            print(f'Reading file {file} ({fidx+1}/{len(files)})...')
            key = '_'
            this_sampledict = {key: [file]}

            # to avoid out-of-memory issues: split in batches
            nevents = read_num_entries(this_sampledict, treename=args.treename, verbose=False)[key][file]
            print(f'Found {nevents} entries in this file.')
            batch_size = 1000 # maybe later add as argument
            batch_start_indices = list(range(0, nevents, batch_size))
            print(f'Found {len(batch_start_indices)} batches of size {batch_size} for this file.')
            batch_scores = []
            for bidx, batch_start_index in enumerate(batch_start_indices):
                print(f'Reading batch {bidx+1}/{len(batch_start_indices)}...')

                events = read_sampledict(this_sampledict,
                          treename=args.treename,
                          branches=branches_to_read,
                          entry_start=batch_start_index,
                          entry_stop=(batch_start_index+batch_size),
                          verbose=False
                )
                '''
                print("Loaded fields:")
                print(sorted(events[key].fields))
                print("Has SecondaryVertices_mass?", "SecondaryVertices_mass" in events[key].fields)
                '''
                # do object selection
                # note: put before adding new variables for speed,
                # but assumes that the selection does not depend on new variables.
                if objectselection is not None:
                    print('Doing object selection...')
                    events[key] = apply_objectselection(events[key], objectselection[0], objectselection[1])

                # add new variables
                print('Adding new variables...')
                add_variables(events[key])

                # do inference
                scores = infer_events(events[key], args.model, prepdict,
                            translation=translation,
                            do_add_variables=False,
                            batch_size=1000)
                batch_scores.append(scores)

                for score_name, score_array in scores.items():
                    events[key][score_name] = score_array

                '''
                print(type(scores))
                for k, v in scores.items():
                    print(k, type(v), getattr(v, "shape", None))

                print(type(events[key]))
                print(events[key].fields)
                print(ak.num(events[key]["Jets_pt"]))
                '''

                safe_fields = []
                for field in events[key].fields:
                    if field.startswith("JetsConstituents_"):
                        continue
                    if field.startswith("SecondaryVertices_"):
                        continue
                    if field.startswith("V0Candidates_"):
                        continue
                    safe_fields.append(field)

                outdict = {field: events[key][field] for field in safe_fields}

                #outdict = {field: events[key][field] for field in events[key].fields}
                batch_outputfile = os.path.join(
                args.outputdir,
                os.path.basename(file).replace('.root', f'_batch{bidx}.root')
                )
                with uproot.recreate(batch_outputfile) as fout:
                    fout[args.treename] = outdict

            # concatenate batches
            scores = {score_name: np.concatenate([batch[score_name] for batch in batch_scores])
                      for score_name in batch_scores[0].keys()}

            # write to output file
            outputfile = file.replace('/','').replace('.root', '.pkl')
            outputfile = os.path.join(args.outputdir, outputfile)
            if not os.path.exists(args.outputdir): os.makedirs(args.outputdir)
            with open(outputfile, 'wb') as f:
                pickle.dump(scores, f)
