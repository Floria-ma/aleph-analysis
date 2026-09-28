
import os
import sys
import glob
import uproot
import argparse
import numpy as np
from fractions import Fraction
import awkward as ak
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


def delta_r(eta1, phi1, eta2, phi2):
    # eta-phi DeltaR, with dphi wrapped to [-pi, pi]
    dphi = np.arctan2(np.sin(phi1 - phi2), np.cos(phi1 - phi2))
    return np.sqrt((eta1 - eta2)**2 + dphi**2)


def jet_match_dR(events):
    # per-jet dR between reco jet and matched gen quark (jagged, same structure as Jets_eta)
    match_eta = -np.log(np.tan(ak.values_astype(events['Jets_match_theta'], np.float64) / 2.))
    return delta_r(ak.values_astype(events['Jets_eta'], np.float64),
                   ak.values_astype(events['Jets_phi'], np.float64),
                   match_eta,
                   ak.values_astype(events['Jets_match_phi'], np.float64))


def quark_label(pdgid, charge):
    # e.g. pdgId -5 -> '$\bar{b}$ (+1/3)'
    name = {1: 'd', 2: 'u', 3: 's', 4: 'c', 5: 'b', 6: 't'}.get(abs(pdgid), str(abs(pdgid)))
    if pdgid < 0: name = r'\bar{' + name + '}'
    charge = Fraction(float(charge)).limit_denominator(3)
    return f'${name}$ ({"+" if charge > 0 else ""}{charge})'


def make_event_display(event, dR, outputfile):
    # 3D display of the reco jet and matched gen quark momenta, starting from the origin
    # (solid lines: reco jets; dashed lines: gen quarks; same color: matched pair)
    # note: plotted as (pz, px, py) (cyclic permutation, so still right-handed)
    #       and viewed such that y points up, z to the lower left and x to the lower right
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(projection='3d')
    pmax = 0.
    for idx in range(len(dR)):
        color = f'C{idx}'
        # reco jet
        p = np.array([event['Jets_pz'][idx], event['Jets_px'][idx], event['Jets_py'][idx]])
        pmag = np.linalg.norm(p)
        label = f'Reco jet {idx}: |p| = {pmag:.1f} GeV, $\\Delta R$ = {dR[idx]:.3f}'
        ax.plot([0, p[0]], [0, p[1]], [0, p[2]], color=color, linewidth=2.5, label=label)
        ax.text(*(1.05 * p), f'jet {idx}', color=color)
        pmax = max(pmax, pmag)
        # matched gen quark
        theta, phi = event['Jets_match_theta'][idx], event['Jets_match_phi'][idx]
        pmag = event['Jets_match_pt'][idx] / np.sin(theta)
        p = pmag * np.array([np.cos(theta), np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi)])
        qlabel = quark_label(event['Jets_match_pdgId'][idx], event['Jets_match_charge'][idx])
        label = f'Gen quark {idx}: {qlabel}, |p| = {pmag:.1f} GeV'
        ax.plot([0, p[0]], [0, p[1]], [0, p[2]], color=color, linewidth=2.5, linestyle='--', label=label)
        ax.text(*(1.05 * p), qlabel, color=color)
        pmax = max(pmax, pmag)
    # beam axis
    lim = 1.15 * pmax
    ax.plot([-lim, lim], [0, 0], [0, 0], color='grey', linestyle=':', label='Beam axis (z)')
    ax.scatter([0], [0], [0], color='black', s=10)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)
    ax.set_box_aspect((1, 1, 1))
    ax.set_xlabel('$p_z$ [GeV]')
    ax.set_ylabel('$p_x$ [GeV]')
    ax.set_zlabel('$p_y$ [GeV]')
    ax.view_init(elev=20, azim=60)
    ax.set_title(f'Run {event["runNumber"]}, event {event["eventNumber"]}')
    fig.legend(loc='lower center', ncol=2, fontsize='small')
    # small reference frame in the lower left corner (same view angle as the main plot),
    # showing the physical x, y, z directions in plotted (pz, px, py) coordinates
    axref = fig.add_axes([0.08, 0.12, 0.16, 0.16], projection='3d')
    axref.view_init(elev=ax.elev, azim=ax.azim)
    for label, direction, ha in [('x', (0, 1, 0), 'left'), ('y', (0, 0, 1), 'center'), ('z (beam)', (1, 0, 0), 'right')]:
        axref.quiver(0, 0, 0, *direction, color='black', arrow_length_ratio=0.2, linewidth=1)
        axref.text(*(1.3 * np.array(direction)), label, fontsize='small', ha=ha, va='center')
    axref.set_xlim(-0.2, 1)
    axref.set_ylim(-0.2, 1)
    axref.set_zlim(-0.2, 1)
    axref.set_box_aspect((1, 1, 1))
    axref.set_axis_off()
    fig.savefig(outputfile)
    plt.close(fig)


if __name__=='__main__':

    # read command line args
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--inputfiles', nargs='+',
      default=['/eos/user/z/zima/aleph-ntuple/charge_0921_dR/eventlevel/output_qqb_*.root'])
    parser.add_argument('-t', '--treename', default='events')
    parser.add_argument('-o', '--outputdir', default='output_jet_match_dR')
    parser.add_argument('-n', '--nfiles', default=None, type=int)
    parser.add_argument('--xhigh', default=4, type=float)
    parser.add_argument('--nbins', default=64, type=int)
    parser.add_argument('--pthigh', default=60, type=float)
    parser.add_argument('--ptnbins', default=60, type=int)
    parser.add_argument('--delow', default=-50, type=float)
    parser.add_argument('--dehigh', default=30, type=float)
    parser.add_argument('--denbins', default=80, type=int)
    parser.add_argument('--y23cut', default=0.13, type=float,
      help='y23 cut used in the event selection (only for drawing)')
    parser.add_argument('--display_threshold', default=0.5, type=float,
      help='Make event displays for events with at least one jet with dR above this value')
    parser.add_argument('--ndisplays', default=10, type=int,
      help='Maximum number of event displays to make')
    parser.add_argument('--display_max_abscostheta', default=0.9, type=float,
      help='Only make event displays for events where all jets and matched quarks have |cos(theta)| below this value'
           + ' (to avoid large eta-phi dR values from small angular differences close to the beam axis)')
    args = parser.parse_args()

    # find input files
    inputfiles = sorted(set(sum([glob.glob(f) for f in args.inputfiles], [])))
    if args.nfiles is not None: inputfiles = inputfiles[:args.nfiles]
    if len(inputfiles)==0: raise Exception(f'No input files found for {args.inputfiles}.')
    print(f'Found {len(inputfiles)} input files.')

    # loop over files and fill histogram
    # (per file, keeping only the flattened dR values in memory)
    branches = ['Jets_eta', 'Jets_phi', 'Jets_pt', 'Jets_e', 'Event_dmerge2', 'Jets_match_theta', 'Jets_match_phi', 'Jets_match_dR']
    display_branches = ['runNumber', 'eventNumber', 'Jets_theta', 'Jets_px', 'Jets_py', 'Jets_pz',
      'Jets_match_pt', 'Jets_match_pdgId', 'Jets_match_charge']
    bins = np.linspace(0, args.xhigh, args.nbins+1)
    counts = np.zeros(args.nbins)
    ptbins = np.linspace(0, args.pthigh, args.ptnbins+1)
    counts2d = np.zeros((args.nbins, args.ptnbins))
    counts2d_p = np.zeros((args.nbins, args.ptnbins))
    debins = np.linspace(args.delow, args.dehigh, args.denbins+1)
    counts2d_de = np.zeros((args.nbins, args.denbins))
    logy23bins = np.linspace(-5, 0, 51)
    counts2d_y23 = np.zeros((args.nbins, len(logy23bins)-1))
    y23bins = np.linspace(0, 0.42, 85)
    counts2d_y23_linear = np.zeros((args.nbins, len(y23bins)-1))
    thetabins = np.linspace(0, np.pi, 64+1)
    counts2d_theta = np.zeros((args.nbins, len(thetabins)-1))
    counts2d_match_theta = np.zeros((args.nbins, len(thetabins)-1))
    etabins = np.linspace(-4, 4, 80+1)
    counts2d_eta = np.zeros((args.nbins, len(etabins)-1))
    counts2d_match_eta = np.zeros((args.nbins, len(etabins)-1))
    noverflow = 0
    njets = 0
    maxdiff = 0.
    nagree = 0
    dR_values = []
    displays = []
    for idx, inputfile in enumerate(inputfiles):
        print(f'Reading file {idx+1} / {len(inputfiles)}: {inputfile}...')
        with uproot.open(inputfile) as f:
            events = f[args.treename].arrays(branches + display_branches)
        dR_jagged = jet_match_dR(events)
        dR = ak.to_numpy(ak.flatten(dR_jagged)).astype(float)
        dR_stored = ak.to_numpy(ak.flatten(events['Jets_match_dR'])).astype(float)
        maxdiff = max(maxdiff, np.max(np.abs(dR - dR_stored), initial=0.))
        nagree += np.sum(np.abs(dR - dR_stored) < 1e-4)
        counts += np.histogram(dR, bins=bins)[0]
        pt = ak.to_numpy(ak.flatten(events['Jets_pt'])).astype(float)
        counts2d += np.histogram2d(dR, pt, bins=[bins, ptbins])[0]
        pmag = ak.to_numpy(ak.flatten(np.sqrt(events['Jets_px']**2 + events['Jets_py']**2 + events['Jets_pz']**2))).astype(float)
        counts2d_p += np.histogram2d(dR, pmag, bins=[bins, ptbins])[0]
        # energy difference between reco jet and matched gen quark
        # (quark energy approximated by its momentum |p| = pT / sin(theta), i.e. neglecting the quark mass)
        quark_e = events['Jets_match_pt'] / np.sin(events['Jets_match_theta'])
        de = ak.to_numpy(ak.flatten(events['Jets_e'] - quark_e)).astype(float)
        recojet_theta = ak.to_numpy(ak.flatten(events['Jets_theta'])).astype(float)
        quark_theta = ak.to_numpy(ak.flatten(events['Jets_match_theta'])).astype(float)
        counts2d_de += np.histogram2d(dR, de, bins=[bins, debins])[0]
        # y23 = d23 / s, with d23 the exclusive 3 -> 2 jet merging distance (same definition as in the selection),
        # broadcast from event level to jet level
        y23 = ak.broadcast_arrays(events['Event_dmerge2'] / np.square(91.2), events['Jets_pt'])[0]
        logy23 = np.log10(np.maximum(ak.to_numpy(ak.flatten(y23)).astype(float), 1e-10))
        counts2d_y23 += np.histogram2d(dR, logy23, bins=[bins, logy23bins])[0]
        counts2d_y23_linear += np.histogram2d(dR, ak.to_numpy(ak.flatten(y23)).astype(float), bins=[bins, y23bins])[0]
        theta = ak.to_numpy(ak.flatten(events['Jets_theta'])).astype(float)
        counts2d_theta += np.histogram2d(dR, theta, bins=[bins, thetabins])[0]
        match_theta = ak.to_numpy(ak.flatten(events['Jets_match_theta'])).astype(float)
        counts2d_match_theta += np.histogram2d(dR, match_theta, bins=[bins, thetabins])[0]
        eta = ak.to_numpy(ak.flatten(events['Jets_eta'])).astype(float)
        counts2d_eta += np.histogram2d(dR, eta, bins=[bins, etabins])[0]
        match_eta = -np.log(np.tan(match_theta / 2.))
        counts2d_match_eta += np.histogram2d(dR, match_eta, bins=[bins, etabins])[0]
        noverflow += np.sum(dR >= args.xhigh)
        njets += len(dR)
        dR_values.append(dR)
        # keep events with large dR for event displays
        if len(displays) < args.ndisplays:
            mask = ak.fill_none(ak.max(dR_jagged, axis=1) > args.display_threshold, False)
            central = ((np.abs(np.cos(events['Jets_theta'])) < args.display_max_abscostheta)
                       & (np.abs(np.cos(events['Jets_match_theta'])) < args.display_max_abscostheta))
            mask = mask & ak.all(central, axis=1)
            nkeep = args.ndisplays - len(displays)
            for event, dR_event in zip(ak.to_list(events[mask][:nkeep]), ak.to_list(dR_jagged[mask][:nkeep])):
                # run and event number are stored as length-1 vectors
                for key in ['runNumber', 'eventNumber']:
                    if isinstance(event[key], list): event[key] = event[key][0]
                displays.append((idx, event, dR_event))

    # print summary
    all_dR = np.concatenate(dR_values)
    print(f'Number of jets: {njets}')
    print(f'Median dR: {np.median(all_dR):.4f}')
    for threshold in [0.1, 0.2, 0.3, 0.5]:
        print(f'Fraction of jets with dR < {threshold}: {np.mean(all_dR < threshold):.4f}')
    print(f'Fraction of jets with dR >= {args.xhigh} (overflow): {noverflow / njets:.4f}')
    print(f'Fraction of jets with |dR (calculated) - Jets_match_dR (stored)| < 1e-4: {nagree / njets:.4f}'
          + f' (maximum difference: {maxdiff:.2e})')
    if nagree < njets:
        print('WARNING: calculated dR does not agree with stored Jets_match_dR.')

    # make plots
    os.makedirs(args.outputdir, exist_ok=True)
    for logscale in [False, True]:
        fig, ax = plt.subplots()
        ax.stairs(counts, bins, color='tab:blue', label=f'{njets} jets')
        ax.set_xlabel(r'$\Delta R$(reco jet, matched gen quark)')
        ax.set_ylabel('Jets')
        ax.set_xlim(bins[0], bins[-1])
        if logscale: ax.set_yscale('log')
        ax.legend()
        outputfile = os.path.join(args.outputdir, 'jet_match_dR' + ('_log' if logscale else '') + '.png')
        fig.savefig(outputfile)
        plt.close(fig)
        print(f'Saved figure {outputfile}.')

    # make 2D histograms of dR vs jet pT, jet momentum, jet-quark energy difference, and y23
    for hist, ybins, ylabel, tag in [
            (counts2d, ptbins, r'Reco jet $p_T$ [GeV]', 'pt'),
            (counts2d_p, ptbins, r'Reco jet $|\vec{p}|$ [GeV]', 'p'),
            (counts2d_de, debins, r'$E$(reco jet) $-$ $E$(matched gen quark) [GeV]', 'de'),
            (counts2d_y23, logy23bins, r'$\log_{10}(y_{23})$', 'y23'),
            (counts2d_y23_linear, y23bins, r'$y_{23}$', 'y23_linear'),
            (counts2d_theta, thetabins, r'Reco jet $\theta$ [rad]', 'theta'),
            (counts2d_match_theta, thetabins, r'Matched gen quark $\theta$ [rad]', 'match_theta'),
            (counts2d_eta, etabins, r'Reco jet $\eta$', 'eta'),
            (counts2d_match_eta, etabins, r'Matched gen quark $\eta$', 'match_eta')]:
        fig, ax = plt.subplots()
        mesh = ax.pcolormesh(bins, ybins, hist.T, norm=LogNorm(vmin=1), cmap='viridis')
        fig.colorbar(mesh, ax=ax, label='Jets')
        ax.set_xlabel(r'$\Delta R$(reco jet, matched gen quark)')
        ax.set_ylabel(ylabel)
        ax.set_title(f'{njets} jets')
        if tag in ['y23', 'y23_linear']:
            ax.axhline(np.log10(args.y23cut) if tag=='y23' else args.y23cut, color='red', linestyle='--',
              label=f'Selection: $y_{{23}}$ < {args.y23cut}')
            ax.legend(loc='lower right' if tag=='y23' else 'upper right')
        outputfile = os.path.join(args.outputdir, f'jet_match_dR_vs_{tag}.png')
        fig.savefig(outputfile)
        plt.close(fig)
        print(f'Saved figure {outputfile}.')

    # make event displays
    displaydir = os.path.join(args.outputdir, 'event_displays')
    os.makedirs(displaydir, exist_ok=True)
    print(f'Making {len(displays)} event displays for events with dR > {args.display_threshold}'
          + f' and |cos(theta)| < {args.display_max_abscostheta}...')
    for fileidx, event, dR in displays:
        outputfile = os.path.join(displaydir,
          f'event_file{fileidx}_run{event["runNumber"]}_evt{event["eventNumber"]}.png')
        make_event_display(event, dR, outputfile)
        print(f'Saved figure {outputfile}.')
