# Plot analysis results

import os
import sys
import json
import copy
import pickle
import uproot
import argparse
import numpy as np
import awkward as ak
from fnmatch import fnmatch
import matplotlib.pyplot as plt

thisdir = os.path.abspath(os.path.dirname(__file__))
topdir = os.path.abspath(os.path.join(thisdir, '../'))
sys.path.append(topdir)

from tools.variabletools import read_variables
from tools.variabletools import HistogramVariable, DoubleHistogramVariable
from tools.samplelisttools import read_samplelist, read_sampledict, find_files
from tools.lumitools import get_lumidict, get_sqrtsdict
from tools.plottools import make_hist_from_events
from tools.plottools import merge_events, merge_sampledict
from tools.plottools import make_batches
from tools.processinfo import ProcessInfoCollection, ProcessCollection
from analysis.eventselection import load_eventselection, get_variable_names
from analysis.eventselection import eval_expression
from analysis.eventselection import get_selection_mask, get_selection_masks
from analysis.objectselection import load_objectselection
from analysis.objectselection import apply_objectselection
from analysis.systematics import get_weight_variation
from analysis.systematics import format_systematic_name
from analysis.external_variables import read_external_variables
from analysis.external_variables import find_external_files
from analysis.thrust import add_thrust_variables
from plotting.plot import plot

# global pyplot settings
#plt.rc("text", usetex=True)
#plt.rc("font", family="serif")


def make_histograms(datastruct, variables,
        treename = 'events',
        branches_to_read = None,
        files_per_batch = None,
        objectselection = None,
        eventselection = None,
        select_processes = None,
        blinding = None,
        blind_processes = None,
        regions = None,
        recalculate_regions = False,
        external_variables = None,
        splitdict = None,
        weights = None,
        weight_variations = None,
        xsections = None,
        lumi = None):
    '''
    Helper function for making histograms from a given set of sampledicts.
    Input arguments:
    - datastruct: dictionary of the form {data type: sampledict, ...},
      with each sampledict of the form {process name: data},
      with data either a list of files to read, or an awkward array of already loaded events.
    - variables: list of variables to produce histograms for
    Returns:
    - dictionary with the following structure:
      dtype -> region_variable_key -> process_key -> systematic_key -> (counts, errors)
    Note: the systematic_key for nominal is "nominal".
    Note: only systematics in the form of weight variations are supported so far,
          not yet systematics with object variations.
    '''

    # add nominal to weight variations (if not yet provided)
    if weight_variations is None: weight_variations = {}
    weight_variations['nominal'] = None

    # loop over samples
    hists = {}
    for dtype, sampledict in datastruct.items():
        if sampledict is None: continue
        hists[dtype] = {}
        for process_key, files in sampledict.items():
            print(f'Now running on sample {process_key}...')
            do_read_events = True

            # case 1 (default): "files" is a list of files
            if isinstance(files, list): pass
            # case 2: datastruct contains already loaded events from an earlier stage
            elif isinstance(files, ak.highlevel.Array): do_read_events = False
            else:
                msg = f'Data type of sample list for sample {process_key} not recognized:'
                msg += f' {type(files)} ({files}).'
                raise Exception(msg)

            # set weight variations for this sample
            # todo: make more flexible and robust against name changes
            this_weight_variations = copy.deepcopy(weight_variations)
            if dtype=='data': this_weight_variations = {'nominal': None}
            if process_key=='syndata': this_weight_variations = {'nominal': None}
            if process_key=='bkgmodel':
                this_weight_variations = {key: val for key, val in this_weight_variations.items()
                                         if( key=='nominal' or key.startswith('abcdWeight') )}
            else:
                this_weight_variations = {key: val for key, val in this_weight_variations.items()
                                         if( key=='nominal' or not key.startswith('abcdWeight') )}

            # set sample dict and branches to read
            this_sampledict = None
            this_branches_to_read = None
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
            if do_read_events:
                this_sampledict = {process_key: files}
                this_branches_to_read = branches_to_read[:]
                if weights is not None and process_key in weights.keys():
                    for weight_expression in weights[process_key]:
                        this_branches_to_read += get_variable_names(weight_expression)
                for weight_variation, branches in this_weight_variations.items():
                    if weight_variation == 'nominal': continue
                    if branches is None: continue
                    for branch in branches: this_branches_to_read.append(branch)
                # remove duplicates
                this_branches_to_read += extra_branches
                this_branches_to_read = list(set(this_branches_to_read))

            # split files in batches if requested
            # (sometimes needed to not run out of memory...)
            this_batches = [this_sampledict] # default case of no splitting in batches
            if files_per_batch is not None:
                if not do_read_events:
                    msg = 'Incompatible settings used: cannot specify batches if events are already loaded.'
                    raise Exception(msg)
                files_per_batch = int(files_per_batch)
                batched_files = make_batches(files, batch_size=files_per_batch)
                this_batches = [{process_key: batch} for batch in batched_files]
                print(f'Split sample in {len(this_batches)} batches.')

            # loop over batches
            for batch_idx, batch_sampledict in enumerate(this_batches):
                print(f'Reading batch {batch_idx+1} / {len(this_batches)}...')
                if do_read_events:

                    # temp: skip files that do not have their corresponding external variable file.
                    # in most cases, an error should be thrown in this case, but suppress for quick tests
                    #if external_variables is not None:
                    #    external_file_dict = find_external_files(batch_sampledict[process_key], external_variables, verbose=True)
                    #    batch_sampledict[process_key] = [k for k, v in external_file_dict.items() if v is not None]

                    # read events
                    events = read_sampledict(batch_sampledict, treename=treename,
                               branches=this_branches_to_read, verbose=False)
                    events[process_key] = add_thrust_variables(events[process_key])
                    print(sorted(events[process_key].fields))

                else: events = {process_key: files}
                print(f'Read batch with {len(events[process_key])} entries'
                        + f' and {len(events[process_key].fields)} branches.')

                # read external variables
                if external_variables is not None:
                    print(f'Reading external variables from {external_variables}...')
                    external_vars = read_external_variables(
                                      batch_sampledict[process_key],
                                      external_variables
                                    )
                    # add to events
                    for key, val in external_vars.items():
                        events[process_key][key] = val

                # store number of events before any selection (for normalization later)
                nevents = {process_key: len(events[process_key])}

                # do extra object selection
                if objectselection is not None:
                    print('Doing extra object selection...')
                    if isinstance(objectselection, list): pass
                    if isinstance(objectselection, tuple): objectselection = [objectselection]
                    for this_objectselection in objectselection: 
                        events[process_key] = apply_objectselection(events[process_key],
                                                this_objectselection[0], 
                                                this_objectselection[1])

                # do extra event selection
                if eventselection is not None:

                    # check if event selection actually needs to be applied
                    # (process dependent)
                    do_selection = True
                    if select_processes is not None and len(select_processes)==0:
                        msg = 'WARNING: found select_processes which is not None but empty;'
                        msg += ' this is ambiguous and will be treated as None.'
                        print(msg)
                        select_processes = None
                    if select_processes is not None and process_key not in select_processes:
                        do_selection = False
                    
                    if do_selection:
                        print('Doing extra event selection...')
                        nbefore = len(events[process_key])
                        mask = get_selection_mask(events[process_key], eventselection)
                        events[process_key] = events[process_key][mask]
                        nselected = len(events[process_key])
                        print(f'Selected {nselected} out of {nbefore} entries.')

                # recalculate regions
                this_regions = regions
                if regions is not None and recalculate_regions:
                    print('Recalculating regions...')
                    for region_name, selection_string in regions.items():
                        mask = get_selection_mask(events[process_key], selection_string)
                        events[process_key][f'mask-{region_name}'] = mask
                    this_regions = {region_name: f'mask-{region_name}' for region_name in regions.keys()}

                # get nominal weights
                nominal_weights = np.ones(len(events[process_key]))
                # general case if weights are already stored as branches in the samples
                if weights is not None and process_key in weights.keys():
                    for weight_expression in weights[process_key]:
                        weight_values = eval_expression(events[process_key], weight_expression).to_numpy().astype(float)
                        nominal_weights = np.multiply(nominal_weights, weight_values)
                # ad-hoc case with provided lumi and cross-section
                elif dtype=='sim':
                    # note: this will not work in bachted mode,
                    # normalization will be done incorrectly if more than 1 batch is used!
                    if xsections is not None and lumi is not None:
                        xsec = xsections[process_key]
                        nominal_weights = lumi * xsec / nevents[process_key]

                # make masks for subprocesses
                subprocess_masks = {process_key: np.ones(len(events[process_key])).astype(bool)}
                if splitdict is not None and process_key in splitdict.keys():
                    print('Making masks for subprocesses...')
                    subprocess_masks = {}
                    for subprocess_key, subprocess_selection in splitdict[process_key].items():
                        mask = get_selection_mask(events[process_key], subprocess_selection).to_numpy().astype(bool)
                        subprocess_masks[subprocess_key] = mask
                        print(f'  - Subprocess {subprocess_key}: {np.sum(mask)} / {len(mask)} entries.')

                # loop over subprocesses
                for subprocess_key, subprocess_mask in subprocess_masks.items():
                    if not subprocess_key in hists[dtype].keys(): hists[dtype][subprocess_key] = {}

                    # loop over systematics with weight variations
                    # (including also nominal case)
                    for weight_variation in this_weight_variations:
                        systematic_key = format_systematic_name(weight_variation)
                        if not systematic_key in hists[dtype][subprocess_key].keys():
                            hists[dtype][subprocess_key][systematic_key] = {}
                        varied_weights = get_weight_variation(events[process_key], weight_variation)
                        varied_weights = np.multiply(nominal_weights, varied_weights)

                        # loop over regions and variables
                        if this_regions is None: this_regions = {'baseline': None}
                        for region_name, region_mask_name in this_regions.items():
                            for variable in variables:
                                #print(f'Making histogram for {subprocess_key}, {weight_variation}, {region_name}, variable {variable.name}...')
                                region_variable_key = f'{region_name}_{variable.name}'

                                # get mask for this region
                                region_mask = np.ones(len(events[process_key])).astype(bool)
                                if region_mask_name is not None: region_mask = events[process_key][region_mask_name]

                                # optional: blind some histograms
                                # note: this is done per variable rather than upfront,
                                #       because the blinding may depend on the variable.
                                if blinding is not None:
                                    if variable.name not in blinding.keys():
                                        msg = f'Blinding was specified but variable {variable.name} not found in the dict.'
                                        raise Exception(msg)
                                    do_selection = ( (blind_processes is None)
                                                   or (len(blind_processes)>0 and process_key in blind_processes) )
                                    if do_selection:
                                        blinding_selection = blinding[variable.name]
                                        blinding_mask = get_selection_mask(events[process_key], blinding_selection)
                                        region_mask = ((region_mask) & (blinding_mask))

                                # make total mask (region and subprocess)
                                total_mask = ((region_mask) & (subprocess_mask))
                                #print(f'Number of entries: {np.sum(total_mask)} / {len(total_mask)}')

                                # make histogram
                                hist = make_hist_from_events(events[process_key], variable,
                                         weights=varied_weights, mask=total_mask, clipmin=0,
                                         verbose=False, flatten=True)

                                # add to dict
                                if region_variable_key in hists[dtype][subprocess_key][systematic_key].keys():
                                    counts, errors = hists[dtype][subprocess_key][systematic_key][region_variable_key]
                                    counts += hist[0]
                                    errors = np.sqrt(np.square(errors) + np.square(hist[1]))
                                    hists[dtype][subprocess_key][systematic_key][region_variable_key] = (counts, errors)
                                else: hists[dtype][subprocess_key][systematic_key][region_variable_key] = hist

                            # end loop over variables
                        # end loop over regions
                    # end loop over weight variations
                # end loop over subprocesses
            # end loop over batches
        # end loop over processes
    # end loop over dtypes

    # restructure dictionary to put the region/variable outside the processes and systematics
    newhists = {}
    for dtype in hists.keys():
        newhists[dtype] = {}
        processes = list(hists[dtype].keys())
        regvars = list(hists[dtype][processes[0]]['nominal'].keys())
        for regvar in regvars:
            newhists[dtype][regvar] = {}
            for process in processes:
                newhists[dtype][regvar][process] = {}
                for sysvar in hists[dtype][process].keys():
                    newhists[dtype][regvar][process][sysvar] = hists[dtype][process][sysvar][regvar]
    hists = newhists

    # return result
    return hists


def make_events(dtypedict,
        treename='events',
        branches_to_read = None,
        objectselection = None,
        eventselection = None,
        select_processes = None,
        regions = None,
        recalculate_regions = False,
        external_variables = None,
        xsections = None,
        lumi = None):
    '''
    Helper function for getting events from a given set of sampledicts.
    Similar to make_histograms, but simplified in the sense that no binning is performed.
    '''

    # loop over samples
    events = {}
    for dtype, sampledict in dtypedict.items():
        events[dtype] = {}
        if sampledict is None: continue
        for process_key, files in sampledict.items():
            print(f'Now running on sample {process_key}...')

            # temp: skip files that do not have their corresponding external variable file.
            # in most cases, an error should be thrown in this case, but suppress for quick tests
            #if external_variables is not None:
            #    external_file_dict = find_external_files(files, external_variables, verbose=True)
            #    files = [k for k, v in external_file_dict.items() if v is not None]

            # read events
            this_sampledict = {process_key: files}
            print(f'Reading events...')
            events[dtype][process_key] = read_sampledict(this_sampledict, treename=treename,
                                           branches=branches_to_read, verbose=False)[process_key]
            events[dtype][process_key] = add_thrust_variables(events[dtype][process_key])
            nevents = len(events[dtype][process_key])
            nbranches = len(events[dtype][process_key].fields)
            print(f'Read sample with {nevents} entries and {nbranches} branches.')

            # read external variables
            if external_variables is not None:
                print(f'Reading external variables from {external_variables}...')
                external_vars = read_external_variables(
                                  this_sampledict[process_key],
                                  external_variables
                                )
                # add to events
                for key, val in external_vars.items():
                    events[dtype][process_key][key] = val

            # do extra object selection
            if objectselection is not None:
                print('Doing extra object selection...')
                if isinstance(objectselection, list): pass
                if isinstance(objectselection, tuple): objectselection = [objectselection]
                for this_objectselection in objectselection:
                    events[dtype][process_key] = apply_objectselection(events[dtype][process_key],
                                                    this_objectselection[0],
                                                    this_objectselection[1])

            # do extra event selection
            if eventselection is not None:

                # check if event selection actually needs to be applied
                # (process dependent)
                do_selection = True
                if select_processes is not None and len(select_processes)==0:
                    msg = 'WARNING: found select_processes which is not None but empty;'
                    msg += ' this is ambiguous and will be treated as None.'
                    print(msg)
                    select_processes = None
                if select_processes is not None and process_key not in select_processes:
                    do_selection = False

                if do_selection:
                    print('Doing extra event selection...')
                    norig = len(events[dtype][process_key])
                    mask = get_selection_mask(events[dtype][process_key], eventselection)
                    events[dtype][process_key] = events[dtype][process_key][mask]
                    nselected = len(events[dtype][process_key])
                    print(f'Selected {nselected} out of {norig} entries.')

            # recalculate regions
            if regions is not None and recalculate_regions:
                print('Recalculating regions...')
                for region_name, selection_string in regions.items():
                    mask = get_selection_mask(events[dtype][process_key], selection_string)
                    events[dtype][process_key][f'mask-{region_name}'] = mask

            # add cross-section weight if requested
            # (need to do here rather than later in make_histograms,
            #  because the number of events before selections is needed for the normalization factor)
            if dtype=='sim' and xsections is not None and lumi is not None:
                print('Adding cross-section weights to events in branch named "weight"...')
                xsection_weights = np.ones(len(events[dtype][process_key]))
                xsec = xsections[process_key]
                weights = np.ones(len(events[dtype][process_key])) * (lumi * xsec / nevents)
                if 'weight' in events[dtype][process_key].fields:
                    msg = 'WARNINIG: overwriting existing branch "weight"!'
                    print(msg)
                events[dtype][process_key]['weight'] = weights

        # end loop over processes
    # end loop over dtypes

    # return result
    return events


def plot_hists_default(hists_combined, variables, outputdir,
      regions=None, datatag=None,
      colordict=None, labeldict=None, styledict=None, stacklist=None,
      shapes=False, normalizesim=False, dolog=False,
      extracmstext=None, lumiheader=None, event_selection_name=None, select_processes=None):
    '''
    Default plotting loop
    '''

    # make a list of all simulated processes
    dummykey = list(hists_combined['sim'].keys())[0]
    sim_processes = list(hists_combined['sim'][dummykey].keys())

    # make color dict
    if colordict is None:
        colordict = {}
        colordict['qqb'] = 'grey'
        colordict['light'] = 'grey'
        colordict['uudd'] = 'paleturquoise'
        colordict['ss'] = 'dodgerblue'
        colordict['cc'] = 'slateblue'
        colordict['bb'] = 'darkorchid'

    # make label dict
    if labeldict is None:
        labeldict = {}
        for p in sim_processes:
            labeldict[p] = p
        labeldict['bb'] = r'$b\overline{b}$'
        labeldict['cc'] = r'$c\overline{c}$'
        labeldict['ss'] = r'$s\overline{s}$'
        labeldict['uudd'] = r'$u\overline{u}$, $d\overline{d}$'

    # set histogram styles
    if styledict is None:
        styledict = {}
        for p in sim_processes: styledict[p] = 'fill'
        if shapes:
            for p in sim_processes: styledict[p] = 'step'

    # set histogram stacking
    if stacklist is None:    
        stacklist = [p for p in sim_processes]
        normalize = False
        if shapes:
            stacklist = []
            normalize = True

    # loop over regions and variables
    if regions is None: regions = {'baseline': None}
    for region_name, mask_name in regions.items():
        for variable in variables:
            print(f'Plotting selection {region_name}, variable {variable.name}...')
            region_variable_key = f'{region_name}_{variable.name}'

            # get nominal histograms for simulation
            hists_sim_nominal = {}
            for process_key in hists_combined['sim'][region_variable_key].keys():
                hists_sim_nominal[process_key] = hists_combined['sim'][region_variable_key][process_key]['nominal']

            # get histograms for data
            hists_data = None
            if datatag is not None:
                hists_data = {}
                for process_key in hists_combined['data'][region_variable_key].keys():
                    hists_data[process_key] = hists_combined['data'][region_variable_key][process_key]['nominal']

            # concatenate all histograms in a single array (for later use)
            histarray = [h[0] for h in hists_sim_nominal.values()]
            if hists_data is not None:
                histarray += [h[0] for h in hists_data.values()]
            histarray = np.array(histarray)

            # split off data hist
            data = None
            if hists_data is not None: data = {datatag: hists_data[datatag]}

            # make a ProcessCollection
            hists_sim = {}
            for process_key in hists_combined['sim'][region_variable_key].keys():
                for systematic_key, hist in hists_combined['sim'][region_variable_key][process_key].items():
                    histname = f'{process_key}_{region_variable_key}_{systematic_key}'
                    hists_sim[histname] = hist
            pic = ProcessInfoCollection.fromhistlist(list(hists_sim.keys()), region_variable_key)
            pc = ProcessCollection(pic, hists_sim)
            print(pic)

            # extract the systematic uncertainties (per process)
            systematics = {}
            for process_key in hists_combined['sim'][region_variable_key].keys():
                systematic = pc.get_systematics_rss(processes=[process_key])[0]
                systematics[process_key] = (hists_sim_nominal[process_key][0], systematic)

            # define ratios to plot
            ratios = []
            ratio_yaxtitles = []
            if datatag is not None:
                ratios.append([datatag, stacklist])
                ratio_yaxtitles.append('Data / MC')

            # modify label dict to include the yield per process
            this_labeldict = labeldict.copy()
            print_yield = False # maybe later add as argument
            if print_yield:
                for process_key, hist in hists_sim_nominal.items():
                    old_label = labeldict.get(process_key, None)
                    if old_label is None: continue
                    process_yield = np.sum(hist[0])
                    new_label = old_label + ' ({:.2e})'.format(process_yield)
                    this_labeldict[process_key] = new_label

            # set y-axis title
            yaxtitle = 'Events'
            if variable.variable.startswith('Jets_'): yaxtitle = 'Jets'
            if variable.variable.startswith('JetsConstituents_'): yaxtitle = 'Jet constituents'
            if variable.variable.startswith('SecondaryVertices_'): yaxtitle = 'Vertices'
            if variable.variable.startswith('V0Candidates_'): yaxtitle = '$V^0$ candidates'
            include_binwidth = True # maybe later add as argument
            if include_binwidth:
                if variable.unit is not None and len(variable.unit)>0:
                    bins = variable.bins
                    binwidths = bins[1:] - bins[:-1]
                    unique_binwidths = list(set(binwidths))
                    unique_binwidths = ([unique_binwidths[0]]
                        + [el for el in unique_binwidths[1:] if abs(el-unique_binwidths[0])/unique_binwidths[0] > 1e-6])
                    if len(unique_binwidths)==1:
                        binwidth = unique_binwidths[0]
                        binwidthtxt = '{:.2f}'.format(binwidth)
                        if binwidth.is_integer(): binwidthtxt = str(int(binwidth))
                        yaxtitle += f' / {binwidthtxt} {variable.unit}'
                    else: yaxtitle += ' / Bin'
                else: yaxtitle += ' / Bin'

            # do plotting
            if normalize: yaxtitle += ' (normalized)'
            fig, axs = plot(bkg=hists_sim_nominal,
                       data=data,
                       systematics=systematics,
                       variable=variable,
                       stacklist=stacklist,
                       colordict=colordict,
                       labeldict=this_labeldict,
                       styledict=styledict,
                       multdict=None,
                       normalize=normalize,
                       normalizesim=normalizesim,
                       extracmstext=extracmstext,
                       lumiheader=lumiheader,
                       yaxtitle=yaxtitle,
                       dolegend=False,
                       ratios=ratios,
                       ratio_yaxtitles=ratio_yaxtitles)

            # some more plot aesthetics
            axs[0].set_ylim((0, axs[0].get_ylim()[1]*1.4))
            ncols = 1
            if 'score_isB' in variable.variable: ncols = 3 # dirty hard-coded hack
            axs[0].legend(loc='upper right', fontsize=17, ncols=ncols)
            #if len(regions.keys())>1:
            #    axs[0].text(0.05, 0.9, region_name, ha='left', va='top', fontsize=12,
            #        transform=axs[0].transAxes)
            #if event_selection_name is not None:
            #    label = event_selection_name
            #    if select_processes is not None and len(select_processes)>0:
            #        label += ' (for {})'.format(', '.join(select_processes))
            #    axs[0].text(0.05, 0.85, label, ha='left', va='top', fontsize=12,
            #      transform=axs[0].transAxes)
            if normalizesim:
                axs[0].text(0.05, 0.8, 'Simulation normalized to data', ha='left', va='top', fontsize=15,
                  transform=axs[0].transAxes)
            # data ratio pad
            #if datatag is not None: axs[1].set_ylim((0, 2))

            # dirty hard-coded ad-hoc hack for one particular figure...
            dov0mass = True
            if dov0mass and variable.variable=='V0Candidates_mass':
                axs[0].axvline(x=0.498, linestyle='--', color='red')
                kstext = r'$K^0_S$ (498 MeV)'
                axs[0].text(0.47, 0.95*axs[0].get_ylim()[1], kstext, ha='right', va='top', fontsize=17, color='red')
                axs[0].axvline(x=1.116, linestyle='--', color='red')
                latext = r'$\Lambda^0$ (1.116 GeV)'
                axs[0].text(1.08, 0.95*axs[0].get_ylim()[1], latext, ha='right', va='top', fontsize=17, color='red')

            # save the figure
            fig.tight_layout()
            figname = region_name + '_' + variable.name + '.png'
            figname = os.path.join(outputdir, figname)
            if not os.path.exists(outputdir): os.makedirs(outputdir)
            fig.savefig(figname)
            fig.savefig(figname.replace('.png', '.pdf'))
            plt.close(fig)
            print(f'Figure saved to {figname}.')
            del axs
            del fig

            # same with log scale
            if dolog:
                fig, axs = plot(bkg=hists_sim_nominal,
                       data=data,
                       systematics=systematics,
                       variable=variable,
                       stacklist=stacklist,
                       colordict=colordict,
                       labeldict=this_labeldict,
                       styledict=styledict,
                       logscale=True,
                       multdict=None,
                       normalize=normalize,
                       normalizesim=normalizesim,
                       extracmstext=extracmstext,
                       lumiheader=lumiheader,
                       yaxtitle=yaxtitle,
                       dolegend=False,
                       ratios=ratios,
                       ratio_yaxtitles=ratio_yaxtitles)

                # some more plot aesthetics
                if np.any(histarray > 0):
                    if not normalize: ymin = np.min(histarray[np.nonzero(histarray)])
                    else: ymin = axs[0].get_ylim()[0]
                    axs[0].set_ylim((ymin, axs[0].get_ylim()[1]**1.4))
                axs[0].legend(loc='upper right', fontsize=17, ncols=ncols)
                #if len(regions.keys())>1:
                #    axs[0].text(0.05, 0.9, region_name, ha='left', va='top', fontsize=12,
                #        transform=axs[0].transAxes)
                #if event_selection_name is not None:
                #    label = event_selection_name
                #    if select_processes is not None and len(select_processes)>0:
                #        label += ' (for {})'.format(', '.join(select_processes))
                #    axs[0].text(0.05, 0.85, label, ha='left', va='top', fontsize=12,
                #      transform=axs[0].transAxes)
                if normalizesim:
                    axs[0].text(0.05, 0.8, 'Simulation normalized to data', ha='left', va='top', fontsize=15,
                      transform=axs[0].transAxes)
                # data ratio pad
                #if datatag is not None: axs[1].set_ylim((0, 2))

                # dirty hard-coded ad-hoc hack for one particular figure...
                dov0dxy = True
                if dov0mass and variable.variable=='V0Candidates_dxy':
                    '''axs[0].axvline(x=6.5, linestyle='--', color='red')
                    txt = axs[0].text(8, axs[0].get_ylim()[1]**0.95, 'VDET layer 1', va='top', fontsize=17, color='red')
                    txt.set_bbox(dict(facecolor='white', alpha=0.5, edgecolor='white'))
                    axs[0].axvline(x=11.3, linestyle='--', color='red')
                    txt = axs[0].text(12.8, axs[0].get_ylim()[1]**0.9, 'VDET layer 2', va='top', fontsize=17, color='red')
                    txt.set_bbox(dict(facecolor='white', alpha=0.5, edgecolor='white'))
                    axs[0].axvline(x=16, linestyle='--', color='red')
                    txt = axs[0].text(17.5, axs[0].get_ylim()[1]**0.85, 'ITC inner radius', va='top', fontsize=17, color='red')
                    txt.set_bbox(dict(facecolor='white', alpha=0.5, edgecolor='white'))
                    axs[0].axvline(x=26, linestyle='--', color='red')
                    txt = axs[0].text(27.5, axs[0].get_ylim()[1]**0.8, 'ITC outer radius', va='top', fontsize=17, color='red')
                    txt.set_bbox(dict(facecolor='white', alpha=0.5, edgecolor='white'))'''
                    #axs[0].axvline(x=30, linestyle='--', color='red')
                    #txt = axs[0].text(31.5, axs[0].get_ylim()[1]**0.75, 'TPC inner radius', va='top', fontsize=17, color='red')
                    #txt.set_bbox(dict(facecolor='white', alpha=0.5, edgecolor='white'))

                # save the figure
                fig.tight_layout()
                figname = region_name + '_' + variable.name + '_log.png'
                figname = os.path.join(outputdir, figname)
                fig.savefig(figname)
                fig.savefig(figname.replace('.png', '.pdf'))
                plt.close(fig)
                print(f'Figure saved to {figname}.')
                del axs
                del fig


if __name__=='__main__':

    # read command line args
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--sim', required=True, nargs='+')
    parser.add_argument('-d', '--data', default=None, nargs='+')
    parser.add_argument('-v', '--variables', required=True, nargs='+')
    parser.add_argument('-o', '--outputdir', required=True)
    parser.add_argument('--objectselection', default=None, nargs='+')
    parser.add_argument('--eventselection', default=None)
    parser.add_argument('--select_processes', default=[], nargs='+')
    parser.add_argument('--regions', default=None)
    parser.add_argument('--recalculate_regions', default=False, action='store_true')
    parser.add_argument('--external_variables', default=None)
    parser.add_argument('--files_per_batch', default=None)
    parser.add_argument('--year', default=None)
    parser.add_argument('--luminosity', default=-1, type=float)
    parser.add_argument('--sqrts', default=-1, type=float)
    parser.add_argument('--xsections', default=None)
    parser.add_argument('--merge', default=None)
    parser.add_argument('--split', default=None)
    parser.add_argument('--normalizesim', default=False, action='store_true')
    parser.add_argument('--shapes', default=False, action='store_true')
    parser.add_argument('--dolog', default=False, action='store_true')
    args = parser.parse_args()

    # set weight variations to include in the uncertainty band
    # (hard-coded for now, maybe extend later)
    weight_variations = {}

    # parse arguments
    if args.data is not None and len(args.data)==0: args.data = None

    # read regions
    regions = None
    if args.regions is not None:
        regions = load_eventselection(args.regions)
        if not args.recalculate_regions:
            # if the regions are not to be recalculated,
            # use already existing masks (assumed to be present in input files)
            regions = {s: f'mask-{s}' for s in regions.keys()}
        # also add a region with no additional selection applied
        regions['baseline'] = None
        print('Found following regions:')
        print(list(regions.keys()))

    # read extra object selection to apply
    objectselection = None
    if args.objectselection is not None:
        objectselection = []
        for f in args.objectselection:
            this_objectselection = load_objectselection(f)
            objectselection.append(this_objectselection)
            print('Found following extra object selection to apply:')
            print(this_objectselection[0])
            print('(to the following branches):')
            print(this_objectselection[1])

    # read extra selection to apply
    event_selection_name = None
    eventselection = None
    select_processes = None
    if args.eventselection is not None:
        eventselection = load_eventselection(args.eventselection, nexpect=1)
        print('Found following extra event selection to apply:')
        print(eventselection)
        if len(args.select_processes)>0:
            select_processes = args.select_processes
            print('(selection will be applied only to the following processes:'
                    + f' {select_processes})')
        event_selection_name = list(eventselection.keys())[0]
        eventselection = eventselection[event_selection_name]

    # read cross-sections
    xsections = None
    if args.xsections is not None:
        with open(args.xsections, 'r') as f:
            xsections = json.load(f)
        print('Found following cross-sections:')
        print(json.dumps(xsections, indent=2))

    # read merging instructions
    mergedict = None
    if args.merge is not None:
        with open(args.merge, 'r') as f:
            mergedict = json.load(f)
        print('Found following instructions for merging samples:')
        print(json.dumps(mergedict, indent=2))

    # read splitting instructions
    splitdict = None
    if args.split is not None:
        with open(args.split, 'r') as f:
            splitdict = json.load(f)
        print('Found following instructions for splitting samples:')
        print(json.dumps(splitdict, indent=2))

    # find samples for simulation
    sampledirs_sim = []
    print('Finding sample files for simulation...')
    for sampledir in args.sim:
        # first check if a file 'files.json' is present (i.e. after merging years)
        ffile = os.path.join(sampledir, 'files.json')
        if os.path.exists(ffile): sampledirs_sim.append(ffile)
        # else default case: find all .root files in the given directory
        else: sampledirs_sim.append(sampledir)
    sampledict_sim = find_files(sampledirs_sim, verbose=False)
    #print('Found following sample dict for simulation:')
    #print(json.dumps(sampledict_sim, indent=2))
    nsimfiles = sum([len(v) for v in sampledict_sim.values()])
    print(f'Found {nsimfiles} simulation files.')

    # find samples for data
    sampledict_data = None
    if args.data is not None:
        sampledirs_data = []
        print('Finding sample files for data...')
        for sampledir in args.data:
            # first check if a file 'files.json' is present (i.e. after merging years)
            ffile = os.path.join(sampledir, 'files.json')
            if os.path.exists(ffile): sampledirs_data.append(ffile)
            # else default case: find all .root files in the given directory
            else: sampledirs_data.append(sampledir)
        sampledict_data = find_files(sampledirs_data, verbose=False)
        #print('Found following sample dict for data:')
        #print(json.dumps(sampledict_data, indent=2))
        ndatafiles = sum([len(v) for v in sampledict_data.values()])
        print(f'Found {ndatafiles} data files.')

    # do merging
    if mergedict is not None:
        print('Merging samples...')
        sampledict_sim = merge_sampledict(sampledict_sim, mergedict, verbose=True)
        if sampledict_data is not None:
            print('Merging data...')
            sampledict_data = merge_sampledict(sampledict_data, mergedict, verbose=False)
        # printouts for testing
        print('Number of files for (merged) samples:')
        for sampledict in [sampledict_sim, sampledict_data]:
            if sampledict is None: continue
            for key, val in sampledict.items():
                print(f'  - {key}: {len(val)}')

    # read variables
    variables = sum([read_variables(f) for f in args.variables], [])
    variablelist = []
    for variable in variables:
        if isinstance(variable, DoubleHistogramVariable):
            variablelist.append(variable.primary.variable)
            variablelist.append(variable.secondary.variable)
        else:
            variablelist.append(variable.variable)
    variablelist = sum([get_variable_names(v) for v in variablelist], [])
    variablelist = list(set(variablelist))

    # get luminosity and center-of-mass energy from year
    luminosity = args.luminosity
    sqrts = args.sqrts
    if args.year is not None:
        lumi_from_year = get_lumidict()[args.year]
        sqrts_from_year = get_sqrtsdict()[args.year]
        if args.luminosity is None or args.luminosity < 0:
            luminosity = lumi_from_year
        elif luminosity!=lumi_from_year:
            msg = f'WARNING: found inconsistency between provided luminosity ({luminosity})'
            msg += f' and the one corresponding to the provided year ({args.year}: {lumi_from_year}).'
            print(msg)
        if args.sqrts is None or args.sqrts < 0:
            sqrts = sqrts_from_year
        elif sqrts!=sqrts_from_year:
            msg = f'WARNING: found inconsistency between provided sqrt(s) ({sqrts})'
            msg += f' and the one corresponding to the provided year ({args.year}: {sqrts_from_year}).'
            print(msg)
    if luminosity < 0: luminosity = None
    if sqrts < 0: sqrts = None

    # define variables to read
    branches_to_read = []
    # add masks
    if regions is not None:
        if not args.recalculate_regions:
            for mask_name in regions.values():
                if mask_name is None: continue
                branches_to_read.append(mask_name)
        else:
            for selection_string in regions.values():
                if selection_string is None: continue
                branches_to_read += get_variable_names(selection_string)
    # add selection
    if objectselection is not None:
        for this_objectselection in objectselection:
            branches_to_read += get_variable_names(this_objectselection[0])
    if eventselection is not None:
        branches_to_read += get_variable_names(eventselection)
    # add variables to plot
    branches_to_read += variablelist[:]
    # add variables needed for splitting
    if splitdict is not None:
        for splitkey, this_splitdict in splitdict.items():
            for selection_string in this_splitdict.values():
                branches_to_read += get_variable_names(selection_string)
    # remove potential duplicates
    branches_to_read = list(set(branches_to_read))
    print('Found following branches to read:')
    print(branches_to_read)

    # make histograms
    dtypedict = {'sim': sampledict_sim, 'data': sampledict_data}
    hists_combined = make_histograms(dtypedict, variables,
                       branches_to_read = branches_to_read,
                       files_per_batch = args.files_per_batch,
                       objectselection = objectselection,
                       eventselection = eventselection,
                       select_processes = select_processes,
                       regions = regions,
                       recalculate_regions = args.recalculate_regions,
                       external_variables = args.external_variables,
                       splitdict = splitdict,
                       weight_variations = weight_variations,
                       lumi = luminosity,
                       xsections = xsections)

    # do some parsing after the loop above
    # (now that regions might have been recalculated,
    #  replace their event selection by the corresponding mask)
    if regions is not None:
        regions = {region_name: f'mask-{region_name}' for region_name in regions.keys()}

    # check number of data categories
    # (only one is supported for now)
    datatag = None
    if sampledict_data is not None:
        keys = list(sampledict_data.keys())
        if len(keys)==1: datatag = keys[0]
        else:
            msg = f'Found unexpected number of data categories: {keys}'
            raise Exception(msg)

    # plot aesthetics settings
    extracmstext = 'Archived Data'
    lumiheaderparts = []
    if args.year is not None:
        lumiheaderparts.append(args.year)
    if luminosity is not None:
        lumiheaderparts.append('{:.1f}'.format(luminosity) + ' pb$^{-1}$')
    if sqrts is not None:
        lumiheaderparts.append('{:.1f}'.format(sqrts) + ' GeV')
    lumiheader = ', '.join(lumiheaderparts)

    # make output directory
    if not os.path.exists(args.outputdir): os.makedirs(args.outputdir)

    # plotting loop
    plot_hists_default(hists_combined, variables, args.outputdir,
      regions=regions, datatag=datatag,
      shapes=args.shapes, normalizesim=args.normalizesim, dolog=args.dolog,
      extracmstext=extracmstext, lumiheader=lumiheader,
      event_selection_name=event_selection_name, select_processes=select_processes)
