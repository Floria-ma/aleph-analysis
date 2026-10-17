import os
import sys
import json
import pickle
import argparse
import numpy as np
import awkward as ak
import matplotlib.pyplot as plt

thisdir = os.path.abspath(os.path.dirname(__file__))
topdir = os.path.abspath(os.path.join(thisdir, '../'))
sys.path.append(topdir)

from tools.samplelisttools import find_files
from tools.samplelisttools import read_sampledict
from tools.samplelisttools import read_samplelist
from tools.samplelisttools import read_num_entries
from tools.plottools import merge_sampledict
from evaluation.inferencetools import infer_events, add_variables, make_inference_session
from evaluation.trackingvariations import (
    apply_tracking_variation,
    load_tracking_variations,
    should_apply_tracking_variation,
)
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
    parser.add_argument('--event-batch-size', type=int, default=1000)
    parser.add_argument('--inference-batch-size', type=int, default=1000)
    parser.add_argument('--onnx-threads', type=int, default=8)
    parser.add_argument('--max-events', type=int, default=None)
    parser.add_argument('--tracking-variation-card', default=None)
    parser.add_argument('--tracking-variation', default=None)
    parser.add_argument(
        '--tracking-variation-apply-to',
        choices=['mc', 'data', 'all'],
        default='mc',
        help='Which input files receive tracking variations. Defaults to MC only.',
    )
    parser.add_argument('--tracking-variation-seed', type=int, default=12345)
    args = parser.parse_args()

    # load the preprocess dict
    with open(args.preprocess, 'r') as f:
        prepdict = json.load(f)

    inference_session = make_inference_session(args.model, num_threads=args.onnx_threads)

    tracking_variation_card = None
    if args.tracking_variation is not None:
        if args.tracking_variation_card is None:
            raise SystemExit(
                '--tracking-variation-card is required when --tracking-variation is set.'
            )
        tracking_variation_card = load_tracking_variations(args.tracking_variation_card)

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
            if args.max_events is not None:
                nevents = min(nevents, args.max_events)
            print(f'Found {nevents} entries in this file.')
            batch_size = args.event_batch_size
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

                # do object selection
                # note: put before adding new variables for speed,
                # but assumes that the selection does not depend on new variables.
                if objectselection is not None:
                    print('Doing object selection...')
                    events[key] = apply_objectselection(events[key], objectselection[0], objectselection[1])

                if (
                    tracking_variation_card is not None
                    and should_apply_tracking_variation(file, args.tracking_variation_apply_to)
                ):
                    print(f'Applying tracking variation {args.tracking_variation}...')
                    events[key] = apply_tracking_variation(
                        events[key],
                        tracking_variation_card,
                        args.tracking_variation,
                        file,
                        bidx,
                        seed=args.tracking_variation_seed,
                    )

                # add new variables
                print('Adding new variables...')
                add_variables(events[key])

                # do inference
                scores = infer_events(events[key], args.model, prepdict,
                            translation=translation,
                            do_add_variables=False,
                            session=inference_session,
                            batch_size=args.inference_batch_size)
                batch_scores.append(scores)

            # concatenate batches
            scores = {score_name: np.concatenate([batch[score_name] for batch in batch_scores])
                      for score_name in batch_scores[0].keys()}

            # write to output file
            outputfile = os.path.basename(file).replace('.root', '.pkl')
            outputfile = os.path.join(args.outputdir, outputfile)
            if not os.path.exists(args.outputdir): os.makedirs(args.outputdir)
            with open(outputfile, 'wb') as f:
                pickle.dump(scores, f)
