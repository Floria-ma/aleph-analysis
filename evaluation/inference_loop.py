import os
import sys
import six
import glob

thisdir = os.path.abspath(os.path.dirname(__file__))
topdir = os.path.abspath(os.path.join(thisdir, '../'))
sys.path.append(topdir)

import tools.condortools as ct
import tools.slurmtools as st


if __name__=='__main__':

    # settings
    modeltag = '20260305_withnewks_withdedx_masked_standardized'
    ntupletag = 'withks'
    model = f'/eos/user/l/llambrec/aleph-data/models/output_{modeltag}/model.onnx'
    preprocess = model.replace('model.onnx', 'preprocess.json')
    outputdir = f'/eos/user/z/zima/aleph-data/output_scores_model_{modeltag}'
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

    # find model
    if not os.path.exists(model):
        raise Exception(f'Model {model} does not exist.')

    # filter resubmission
    if resubmit:
        resubmit_files = []
        for inputfile in inputfiles:
            outputfile = inputfile.replace('/','').replace('.root', '_withscores.root')
            outputfile = os.path.join(outputdir, outputfile)
            if not os.path.exists(outputfile):
                resubmit_files.append(inputfile)
        inputfiles = resubmit_files
        print(f'Found {len(inputfiles)} files for resubmission.')
        print(f'Continue? (y/n)')
        go = six.moves.input()
        if go!='y': sys.exit()

    # loop over input files
    cmds = []
    for f in inputfiles:

        # set object selection
        # update: do not do object (i.e. jet) selection anymore,
        #         as we want to store the scores for individual jets as well
        #         rather than per-event combined scores only;
        #         but then the jet selection needs to be applied at a later stage
        #         when making the per-event combined scores!
        #objectselection = 'selections/selection_jets.json'
        objectselection = None

        # make the command
        cmd = 'python inference.py'
        cmd += f' -s {f}'
        cmd += f' -m {model}'
        cmd += f' -p {preprocess}'
        cmd += f' -o {outputdir}'
        cmd += ' -t events'
        if objectselection is not None: cmd += f' --objectselection {objectselection}'
        cmd += f' --translation translations/translations.json'
        cmds.append(cmd)

    # test mode
    if test:
        print('WARNING: test mode is set to True.')
        cmds = [cmds[0]]
        runmode = 'local'

    # run commands
    if runmode=='local':
        for cmd in cmds:
            print(cmd)
            os.system(cmd)
    elif runmode=='condor':
        conda_activate = 'source /afs/cern.ch/user/z/zima/venvs/aleph-infer/bin/activate'
        ct.submitCommandsAsCondorCluster('cjob_inference', cmds,
          jobflavour='workday', conda_activate=conda_activate)
    elif runmode=='slurm':
        env_cmds = ([
          'export PATH=/blue/avery/llambre1.brown/miniforge3/envs/weaver/bin:$PATH',
          f'cd {thisdir}'
        ])
        slurmscript = 'sjob_inference.sh'
        job_name = os.path.splitext(slurmscript)[0]
        st.submitCommandsAsSlurmJobs(cmds, script=slurmscript,
                job_name=job_name, env_cmds=env_cmds,
                memory='16G', time='05:00:00', constraint='el9')
