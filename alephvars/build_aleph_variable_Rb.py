# Build high-level ALEPH variables

import os
import sys
import json
import copy
import scipy
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

from tools.variabletools import read_variables, HistogramVariable, DoubleHistogramVariable
from tools.samplelisttools import read_samplelist, read_sampledict, find_files
from tools.lumitools import get_lumidict
from tools.plottools import merge_events, merge_sampledict
from tools.plottools import make_batches
from tools.processinfo import ProcessInfoCollection, ProcessCollection
from analysis.eventselection import load_eventselection, get_variable_names
from analysis.eventselection import get_selection_mask, get_selection_masks
from analysis.objectselection import load_objectselection
from analysis.objectselection import apply_objectselection
from analysis.plot import make_histograms, make_events
from plotting.plot import plot
from alephvars import ipsig_prob, jet_ipsig_prob, mass_ipsig_prob, thrust_axis


if __name__=='__main__':

    # read command line args
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--sim', required=True, nargs='+')
    parser.add_argument('-d', '--data', default=None, nargs='+')
    parser.add_argument('-v', '--variables', required=True, nargs='+')
    parser.add_argument('-o', '--outputdir', required=True)
    parser.add_argument('--objectselection', default=None)
    parser.add_argument('--eventselection', default=None)
    parser.add_argument('--select_processes', default=[], nargs='+')
    parser.add_argument('--regions', default=None)
    parser.add_argument('--recalculate_regions', default=False, action='store_true')
    parser.add_argument('--files_per_batch', default=None)
    parser.add_argument('--year', default=None)
    parser.add_argument('--luminosity', default=-1, type=float)
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
        objectselection = load_objectselection(args.objectselection)
        print('Found following extra object selection to apply:')
        print(objectselection[0])
        print('(to the following branches):')
        print(objectselection[1])

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

    # add extra variables to read by hand
    variablelist += ([
        'pfcand_pt',
        'pfcand_px',
        'pfcand_py',
        'pfcand_pz',
        'pfcand_e',
        'pfcand_nTrackHits_VDET',
        'pfcand_nTrackHits_TPC',
        'pfcand_trackChi2Normalized',
        'pfcand_dxy',
        'pfcand_dz',
        'pfcand_dxydxy',
        'pfcand_dzdz',
        'pfcand_thetarel',
        'pfcand_linearSignedIP3D',
        'pfcand_transverseJetDistance',
        'pfcand_longitudinalJetDistance',
    ])

    # get luminosity from year
    luminosity = args.luminosity
    if args.year is not None:
        lumi_from_year = get_lumidict()[args.year]
        if args.luminosity is None or args.luminosity < 0:
            luminosity = lumi_from_year
        elif luminosity!=lumi_from_year:
            msg = f'WARNING: found inconsistency between provided luminosity ({luminosity})'
            msg += f' and the one corresponding to the provided year ({args.year}: {lumi_from_year}).'
            print(msg)
    if luminosity < 0: luminosity = None

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
        branches_to_read += get_variable_names(objectselection[0])
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

    # make events
    dtypedict = {'sim': sampledict_sim}
    if sampledict_data is not None: dtypedict['data'] = sampledict_data
    events_combined = make_events(dtypedict,
                       treename = 'tree',
                       branches_to_read = branches_to_read,
                       objectselection = objectselection,
                       eventselection = eventselection,
                       select_processes = select_processes,
                       regions = regions,
                       recalculate_regions = args.recalculate_regions)

    # build variables
    for dtype in events_combined.keys():
        for process_key in events_combined[dtype].keys():
            events = events_combined[dtype][process_key]
            
            # get impact parameter significance
            ipsig2d = events['pfcand_Sip2dSig']
            ipsig3d = events['pfcand_Sip3dSig']
            linipsig3d = events['pfcand_linearSignedIP3DSig']
            charged_mask = (np.abs(ipsig2d-(-9))>1e-12)
            ipsig2d = ipsig2d[charged_mask]
            ipsig3d = ipsig3d[charged_mask]
            linipsig3d = linipsig3d[charged_mask]

            # get auxiliary variables (used in selection)
            vdethits = events['pfcand_nTrackHits_VDET'][charged_mask]
            tpchits = events['pfcand_nTrackHits_TPC'][charged_mask]
            dxy = events['pfcand_dxy'][charged_mask]
            dz = events['pfcand_dz'][charged_mask]
            dxyerr = np.sqrt(events['pfcand_dxydxy'][charged_mask])
            dzerr = np.sqrt(events['pfcand_dzdz'][charged_mask])
            normchi2 = events['pfcand_trackChi2Normalized'][charged_mask]
            pt = events['pfcand_pt'][charged_mask]
            thetarel = events['pfcand_thetarel'][charged_mask]
            linip3d = events['pfcand_linearSignedIP3D'][charged_mask]
            linip3derr = np.divide(linip3d, linipsig3d)
            transjetdist = events['pfcand_transverseJetDistance'][charged_mask]
            lonjetdist = events['pfcand_longitudinalJetDistance'][charged_mask]
           
            # do track filtering
            mask = (
                (vdethits >= 1)
                & (tpchits >= 4)
                & (np.abs(dxy) < 0.5)
                & (np.abs(dz) < 0.5)
                & (dxyerr < 0.1)
                & (dzerr < 0.1)
                & (normchi2 < 5)
                & (pt > 0.4)
                & (np.cos(thetarel) > 0.7)
                & (np.abs(linip3d) < 0.25)
                & (linip3derr < 0.075)
                & (transjetdist < 0.04)
                & (lonjetdist < 1)
            )
            ipsig2d = ipsig2d[mask]
            ipsig3d = ipsig3d[mask]
            linipsig3d = linipsig3d[mask]
            flatmask = ak.flatten(mask).to_numpy()
            print(f'Selected {np.sum(flatmask.astype(int))} out of {len(flatmask)} jet constituents.')
 
            # convert to probability
            ipsig2d_prob = ipsig_prob(ipsig2d)
            ipsig3d_prob = ipsig_prob(ipsig3d)
            linipsig3d_prob = ipsig_prob(linipsig3d)

            # from per-constituent to per-jet variable
            pj2d = jet_ipsig_prob(ipsig2d, prob=ipsig2d_prob)
            pj3d = jet_ipsig_prob(ipsig3d, prob=ipsig3d_prob)
            linpj3d = jet_ipsig_prob(linipsig3d, prob=linipsig3d_prob)

            # take log for easier plotting
            pj2d = np.where(pj2d < 1e-10, 1e-10, pj2d)
            pj3d = np.where(pj3d < 1e-10, 1e-10, pj3d)
            inpj3d = np.where(linpj3d < 1e-10, 1e-10, linpj3d)
            pj2dlog = -np.log10(pj2d)
            pj3dlog = -np.log10(pj3d)
            linpj3dlog = -np.log10(linpj3d)

            # printouts for debugging
            #print(len(pj3d))
            #print(np.amin(pj3d))
            #print(np.amax(pj3d))
            #print(len(pj3dlog))
            #print(np.amin(pj3dlog))
            #print(np.amax(pj3dlog))

            # alternative per-jet variable using invariant mass
            track_vectors = ak.zip({
                "px": events["pfcand_px"][mask],
                "py": events["pfcand_py"][mask],
                "pz": events["pfcand_pz"][mask],
                "e": events["pfcand_e"][mask]})
            linmuj3d = mass_ipsig_prob(linipsig3d, track_vectors, prob=linipsig3d_prob)
            linmuj3d = np.where(linmuj3d < 1e-10, 1e-10, linmuj3d)
            linmuj3dlog = -np.log10(linmuj3d)

            # add variable for plotting
            events['recojet_pj2d'] = pj2d
            events['recojet_pj3d'] = pj3d
            events['recojet_linpj3d'] = linpj3d
            events['recojet_linmuj3d'] = linmuj3d
            events['recojet_pj2dlog'] = pj2dlog
            events['recojet_pj3dlog'] = pj3dlog
            events['recojet_linpj3dlog'] = linpj3dlog
            events['recojet_linmuj3dlog'] = linmuj3dlog
            events['pfcand_Sip2dSigProb'] = ipsig2d_prob
            events['pfcand_Sip3dSigProb'] = ipsig3d_prob
            events['pfcand_linearSignedIP3DSigProb'] = linipsig3d_prob
            events['pfcand_Sip2dSig'] = ipsig2d # overwrite after selection
            events['pfcand_Sip3dSig'] = ipsig3d # overwrite after selection
            events['pfcand_linearSignedIP3DSig'] = linipsig3d # overwrite after selection
            variables.append(
                HistogramVariable.fromdict({
                  'name': 'pj2d',
                  'variable': 'recojet_pj2d',
                  'axtitle': 'PJ2D',
                  'unit': None,
                  'nbins': 100,
                  'xlow': 0,
                  'xhigh': 1
                })
            )
            variables.append(
                HistogramVariable.fromdict({
                  'name': 'pj3d',
                  'variable': 'recojet_pj3d',
                  'axtitle': 'PJ3D',
                  'unit': None,
                  'nbins': 100,
                  'xlow': 0,
                  'xhigh': 1
                })
            )
            variables.append(
                HistogramVariable.fromdict({
                  'name': 'linpj3d',
                  'variable': 'recojet_linpj3d',
                  'axtitle': 'PJ3D (ALEPH-style)',
                  'unit': None,
                  'nbins': 100,
                  'xlow': 0,
                  'xhigh': 1
                })
            )
            variables.append(
                HistogramVariable.fromdict({
                  'name': 'linmuj3d',
                  'variable': 'recojet_linmuj3d',
                  'axtitle': 'MJ3D (ALEPH-style)',
                  'unit': None,
                  'nbins': 100,
                  'xlow': 0,
                  'xhigh': 1
                })
            )
            variables.append(
                HistogramVariable.fromdict({
                  'name': 'pj2dlog',
                  'variable': 'recojet_pj2dlog',
                  'axtitle': '-log(PJ2D)',
                  'unit': None,
                  'nbins': 100,
                  'xlow': 0,
                  'xhigh': 10
                })
            )
            variables.append(
                HistogramVariable.fromdict({
                  'name': 'pj3dlog',
                  'variable': 'recojet_pj3dlog',
                  'axtitle': '-log(PJ3D)',
                  'unit': None,
                  'nbins': 100,
                  'xlow': 0,
                  'xhigh': 10
                })
            )
            variables.append(
                HistogramVariable.fromdict({
                  'name': 'linpj3dlog',
                  'variable': 'recojet_linpj3dlog',
                  'axtitle': '-log(PJ3D) (ALEPH-style)',
                  'unit': None,
                  'nbins': 100,
                  'xlow': 0,
                  'xhigh': 10
                })
            )
            variables.append(
                HistogramVariable.fromdict({
                  'name': 'linmuj3dlog',
                  'variable': 'recojet_linmuj3dlog',
                  'axtitle': '-log(MJ3D) (ALEPH-style)',
                  'unit': None,
                  'nbins': 100,
                  'xlow': 0,
                  'xhigh': 4
                })
            )
            variables.append(
                HistogramVariable.fromdict({
                  'name': 'Sip2dSigProb',
                  'variable': 'pfcand_Sip2dSigProb',
                  'axtitle': 'SIP2D significance probability',
                  'unit': None,
                  'nbins': 100,
                  'xlow': -1,
                  'xhigh': 1
                })
            )
            variables.append(
                HistogramVariable.fromdict({
                  'name': 'Sip3dSigProb',
                  'variable': 'pfcand_Sip3dSigProb',
                  'axtitle': 'SIP3D significance probability',
                  'unit': None,
                  'nbins': 100,
                  'xlow': -1,
                  'xhigh': 1
                })
            )
            variables.append(
                HistogramVariable.fromdict({
                  'name': 'linearSignedIP3DSigProb',
                  'variable': 'pfcand_linearSignedIP3DSigProb',
                  'axtitle': 'SIP3D significance probability (ALEPH-style)',
                  'unit': None,
                  'nbins': 100,
                  'xlow': -1,
                  'xhigh': 1
                })
            )

    # calculate discriminating power
    this_events = events_combined['sim']['qqb']
    for variable in ['recojet_pj2dlog', 'recojet_pj3dlog', 'recojet_linpj3dlog']:
        values = {}
        for subprocess_name, subprocess_selection in splitdict['qqb'].items():
            mask = get_selection_mask(this_events, subprocess_selection).to_numpy().astype(bool)
            this_values = this_events[variable][mask].to_numpy()
            values[subprocess_name] = this_values
        signame = 'bb'
        sigeff = 0.2
        cutoff = np.quantile(values[signame], 1-sigeff)
        effs = {}
        for subprocess_name in splitdict['qqb'].keys():
            effs[subprocess_name] = np.sum((values[subprocess_name] > cutoff).astype(int))/len(values[subprocess_name])
        print(f'Efficiencies for {variable} (target: {signame}, {sigeff})')
        print(effs)

    # make histograms
    hists_combined = make_histograms(events_combined, variables,
                       regions = regions,
                       recalculate_regions = False,
                       splitdict = splitdict,
                       lumi = luminosity,
                       xsections = xsections)

    # do some parsing after the loop above
    # (now that regions might have been recalculated,
    #  replace their event selection by the corresponding mask)
    if regions is not None:
        regions = {region_name: f'mask-{region_name}' for region_name in regions.keys()}

    # some more parsing
    # (make a list of all simulated processes after potential splitting)
    sim_processes = list(dtypedict['sim'].keys())
    if splitdict is not None:
        for key, this_splitdict in splitdict.items():
            vals = list(this_splitdict.keys())
            sim_processes.remove(key)
            sim_processes += vals

    # check number of data categories
    # (only one is supported for now)
    datatag = None
    if sampledict_data is not None:
        keys = list(sampledict_data.keys())
        if len(keys)==1: datatag = keys[0]
        else:
            msg = f'Found unexpected number of data categories: {keys}'
            raise Exception(msg)

    # make color dict
    colordict = {}
    colordict['qqb'] = 'deepskyblue'
    colordict['light'] = 'lightskyblue'
    colordict['cc'] = 'deepskyblue'
    colordict['bb'] = 'darkorchid'

    # make label dict
    labeldict = {}
    for p in sim_processes:
        labeldict[p] = p

    # set histogram styles and stacking
    styledict = {}
    stacklist = []
    # default case: stack all simulation as filled histograms
    for p in sim_processes: styledict[p] = 'fill'
    stacklist = [p for p in sim_processes]
    normalize = False
    # shape comparison mode: no stacking, line histograms, normalized
    if args.shapes:
        for p in sim_processes: styledict[p] = 'step'
        stacklist = []
        normalize = True

    # plot aesthetics settings
    extracmstext = 'Resurrected'
    lumiheaderparts = []
    if args.year is not None:
        lumiheaderparts.append(args.year)
    if luminosity is not None:
        lumiheaderparts.append('{:.2f}'.format(luminosity) + ' pb$^{-1}$')
    lumiheader = ', '.join(lumiheaderparts)

    # settings for multiplying the signal
    multdict = None
    if multdict is not None:
        for key, val in multdict.items():
            if not key in labeldict: continue
            labeldict[key] = labeldict[key] + f' (x {val})'

    # make output directory
    if not os.path.exists(args.outputdir): os.makedirs(args.outputdir)

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
            if datatag is not None: ratios.append([datatag, stacklist])

            # modify label dict to include the yield per process
            this_labeldict = labeldict.copy()
            print_yield = True # maybe later add as argument
            if print_yield:
                for process_key, hist in hists_sim_nominal.items():
                    old_label = labeldict.get(process_key, None)
                    if old_label is None: continue
                    process_yield = np.sum(hist[0])
                    new_label = old_label + ' ({:.2e})'.format(process_yield)
                    this_labeldict[process_key] = new_label

            # do plotting
            yaxtitle = 'Jets'
            if normalize: yaxtitle += ' (normalized)'
            fig, axs = plot(bkg=hists_sim_nominal,
                       data=data,
                       systematics=systematics,
                       variable=variable,
                       stacklist=stacklist,
                       colordict=colordict,
                       labeldict=this_labeldict,
                       styledict=styledict,
                       multdict=multdict,
                       normalize=normalize,
                       normalizesim=args.normalizesim,
                       extracmstext=extracmstext,
                       lumiheader=lumiheader,
                       yaxtitle=yaxtitle,
                       dolegend=False,
                       ratios=ratios)

            # some more plot aesthetics
            axs[0].set_ylim((0, axs[0].get_ylim()[1]*1.4))
            axs[0].legend(loc='upper right', fontsize=12)
            if len(regions.keys())>1:
                axs[0].text(0.05, 0.9, region_name, ha='left', va='top', fontsize=12,
                    transform=axs[0].transAxes)
            if event_selection_name is not None:
                label = event_selection_name
                if select_processes is not None and len(args.select_processes)>0:
                    label += ' (for {})'.format(', '.join(select_processes))
                axs[0].text(0.05, 0.85, label, ha='left', va='top', fontsize=12,
                  transform=axs[0].transAxes)
            if args.normalizesim:
                axs[0].text(0.05, 0.8, 'Simulation normalized to data', ha='left', va='top', fontsize=12,
                  transform=axs[0].transAxes)
            # data ratio pad
            if datatag is not None: axs[1].set_ylim((0, 2))

            # save the figure
            fig.tight_layout()
            figname = region_name + '_' + variable.name + '.png'
            figname = os.path.join(args.outputdir, figname)
            fig.savefig(figname)
            plt.close(fig)
            print(f'Figure saved to {figname}.')
            del axs
            del fig

            # same with log scale
            if args.dolog:
                fig, axs = plot(bkg=hists_sim_nominal,
                       data=data,
                       systematics=systematics,
                       variable=variable,
                       stacklist=stacklist,
                       colordict=colordict,
                       labeldict=this_labeldict,
                       styledict=styledict,
                       logscale=True,
                       multdict=multdict,
                       normalize=normalize,
                       normalizesim=args.normalizesim,
                       extracmstext=extracmstext,
                       lumiheader=lumiheader,
                       yaxtitle=yaxtitle,
                       dolegend=False,
                       ratios=ratios)

                # some more plot aesthetics
                if np.any(histarray > 0):
                    #ymin = np.min(histarray[np.nonzero(histarray)])
                    ymin = axs[0].get_ylim()[0]
                    axs[0].set_ylim((ymin, axs[0].get_ylim()[1]**1.4))
                axs[0].legend(loc='upper right', fontsize=12)
                if len(regions.keys())>1:
                    axs[0].text(0.05, 0.9, region_name, ha='left', va='top', fontsize=12,
                        transform=axs[0].transAxes)
                if event_selection_name is not None:
                    label = event_selection_name
                    if select_processes is not None and len(select_processes)>0:
                        label += ' (for {})'.format(', '.join(select_processes))
                    axs[0].text(0.05, 0.85, label, ha='left', va='top', fontsize=12,
                      transform=axs[0].transAxes)
                if args.normalizesim:
                    axs[0].text(0.05, 0.8, 'Simulation normalized to data', ha='left', va='top', fontsize=12,
                      transform=axs[0].transAxes)
                # data ratio pad
                if datatag is not None: axs[1].set_ylim((0, 2))

                # save the figure
                fig.tight_layout()
                figname = region_name + '_' + variable.name + '_log.png'
                figname = os.path.join(args.outputdir, figname)
                fig.savefig(figname)
                plt.close(fig)
                print(f'Figure saved to {figname}.')
                del axs
                del fig
