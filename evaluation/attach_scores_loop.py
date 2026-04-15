import os
import sys
import six
import glob

thisdir = os.path.abspath(os.path.dirname(__file__))
topdir = os.path.abspath(os.path.join(thisdir, '../'))
sys.path.append(topdir)

import tools.condortools as ct
import tools.slurmtools as st


def root_to_pkl_name(rootfile):
    """
    Map input ROOT path to the corresponding score pickle basename.
    Example:
      /eos/user/l/llambrec/aleph-data/ntuples-withks/eventlevel/data/output_data_99.root
    ->
      eosuserlllambrecaleph-datantuples-withnewkseventleveldataoutput_data_99.pkl
    """
    score_source = rootfile.replace('ntuples-withks', 'ntuples-withnewks')
    return score_source.replace('/', '').replace('.root', '.pkl')


if __name__ == '__main__':

    # settings
    scoretag = '20260305_withnewks_withdedx_masked_standardized'
    ntupletag = 'withks'
    scoredir = f'/eos/user/l/llambrec/aleph-data/model_output_scores/output_scores_model_{scoretag}'
    outputdir = f'/eos/user/z/zima/aleph-data/ntuples_with_attached_scores_{scoretag}'
    runmode = 'condor'
    resubmit = True
    ntuplename = f'ntuples-{ntupletag}' if ntupletag is not None else 'ntuples'
    files = [
        f'/eos/user/l/llambrec/aleph-data/{ntuplename}/eventlevel/mc/output_qqb_*.root',
        f'/eos/user/l/llambrec/aleph-data/{ntuplename}/eventlevel/data/output_data_*.root',
    ]

    # set test mode
    test = False

    # find files
    inputfiles = []
    for pattern in files:
        inputfiles += glob.glob(pattern)
    print(f'Found {len(inputfiles)} files matching patterns.')

    # check matching pickle files exist
    valid_inputfiles = []
    for inputfile in inputfiles:
        pklfile = os.path.join(scoredir, root_to_pkl_name(inputfile))
        if os.path.exists(pklfile):
            valid_inputfiles.append(inputfile)
        else:
            print(f'[WARNING] Missing score file for {inputfile}')
            print(f'          Expected: {pklfile}')

    inputfiles = valid_inputfiles
    print(f'Found {len(inputfiles)} files with matching score pickle.')

    # filter resubmission
    if resubmit:
        resubmit_files = []
        for inputfile in inputfiles:
            outfile = os.path.basename(inputfile).replace('.root', '_withscores.root')
            outfile = os.path.join(outputdir, outfile)
            if not os.path.exists(outfile):
                resubmit_files.append(inputfile)
        inputfiles = resubmit_files
        print(f'Found {len(inputfiles)} files for resubmission.')
        print('Continue? (y/n)')
        go = six.moves.input()
        if go != 'y':
            sys.exit()

    # build commands
    cmds = []
    for f in inputfiles:
        pklfile = os.path.join(scoredir, root_to_pkl_name(f))
        outfile = os.path.basename(f).replace('.root', '_withscores.root')
        outfile = os.path.join(outputdir, outfile)

        cmd = 'python attach_scores.py'
        cmd += f' -i {f}'
        cmd += f' -s {pklfile}'
        cmd += ' -t events'
        cmd += f' -o {outfile}'
        cmds.append(cmd)

    # test mode
    if test:
        print('WARNING: test mode is set to True.')
        cmds = [cmds[0]]
        runmode = 'local'

    # run commands
    if runmode == 'local':
        for cmd in cmds:
            print(cmd)
            os.system(cmd)

    elif runmode == 'condor':
        conda_activate = 'source /afs/cern.ch/user/z/zima/venvs/aleph-infer/bin/activate'
        ct.submitCommandsAsCondorCluster(
            'cjob_attach_scores',
            cmds,
            jobflavour='workday',
            conda_activate=conda_activate
        )

    elif runmode == 'slurm':
        env_cmds = [
            'export PATH=/blue/avery/llambre1.brown/miniforge3/envs/weaver/bin:$PATH',
            f'cd {thisdir}'
        ]
        slurmscript = 'sjob_attach_scores.sh'
        job_name = os.path.splitext(slurmscript)[0]
        st.submitCommandsAsSlurmJobs(
            cmds,
            script=slurmscript,
            job_name=job_name,
            env_cmds=env_cmds,
            memory='16G',
            time='05:00:00',
            constraint='el9'
        )