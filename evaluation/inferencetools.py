import os
import sys
import json
import onnx
import onnxruntime
import numpy as np
import awkward as ak


def _p4_from_ptetaphim(pt, eta, phi, mass):
    # from here: https://github.com/hqucms/weaver-core/blob/9b92955bc89ecada29f5d4e6b9a21f4f0a28f8c5/weaver/utils/data/tools.py#L111
    import vector
    vector.register_awkward()
    return vector.zip({'pt': pt, 'eta': eta, 'phi': phi, 'mass': mass})


def _pad(a, maxlen, value=0, dtype='float32'):
    # from here: https://github.com/hqucms/weaver-core/blob/9b92955bc89ecada29f5d4e6b9a21f4f0a28f8c5/weaver/utils/data/tools.py#L32
    if isinstance(a, np.ndarray) and a.ndim >= 2 and a.shape[1] == maxlen:
        return a
    elif isinstance(a, ak.Array):
        if a.ndim == 1:
            a = ak.unflatten(a, 1)
        a = ak.fill_none(ak.pad_none(a, maxlen, clip=True), value)
        return ak.values_astype(a, dtype)
    else:
        x = (np.ones((len(a), maxlen)) * value).astype(dtype)
        for idx, s in enumerate(a):
            if not len(s):
                continue
            trunc = s[:maxlen].astype(dtype)
            x[idx, :len(trunc)] = trunc
        return x


def _clip(a, a_min, a_max):
    # from here: https://github.com/hqucms/weaver-core/blob/9b92955bc89ecada29f5d4e6b9a21f4f0a28f8c5/weaver/utils/data/tools.py#L61
    if isinstance(a, np.ndarray) or a.ndim == 1:
        return np.clip(a, a_min, a_max)
    else:
        return ak.unflatten(np.clip(ak.to_numpy(ak.flatten(a)), a_min, a_max), ak.num(a))


def add_variables(events, names_only=False):
    '''
    Build new variables and add to events dict.
    Note: hard-coded for now, maybe try to automate later if needed.
    Input arguments:
      - events: events dict of the form {'<variable name>': awkward array, ...}
      - names_only: return only the names of the variables that are needed as input
        and the names of the variables that will be produced,
        without actually producing them.
        In this case the events argument is not used and can be None.
    Returns:
      - If names_only is set to True, returns a dictionary of the form:
        {'input_names': [names of input variables],
         'output_names': [names of variables that will be produced]}
      - Else returns the modified events dict with added variables.
    '''
    if names_only:
        # just return a dict with input and output names
        # (e.g. useful for deciding which branches to read)
        names = {
          'input_names': [
            'JetsConstituents_pt', 'JetsConstituents_e',
            'JetsConstituents_thetarel', 'JetsConstituents_phirel',
            'JetsConstituents_pz',
            'Jets_pt', 'Jets_e', 'Jets_mass',
            'SecondaryVertices_chi2Normalized',
            'SecondaryVertices_p',
            'SecondaryVertices_thetarel',
            'SecondaryVertices_phirel',
            'V0Candidates_chi2Normalized',
            'V0Candidates_p',
            'V0Candidates_thetarel',
            'V0Candidates_phirel',
            'JetsConstituents_charge',
            'JetsConstituents_isChargedHad',
            'JetsConstituents_dEdx_wires_value',
            'JetsConstituents_PID_pval_wires_kaon',
            'JetsConstituents_PID_pval_wires_pi'
          ],
          'output_names': [
            'JetsConstituents_mask',
            'JetsConstituents_pt_log', 'JetsConstituents_e_log',
            'JetsConstituents_drrel',
            'Jets_pt_log', 'Jets_e_log', 'Jets_mass_log',
            'SecondaryVertices_mask',
            'SecondaryVertices_pt_proxy',
            'SecondaryVertices_px_proxy',
            'SecondaryVertices_py_proxy',
            'SecondaryVertices_pz_proxy',
            'SecondaryVertices_e_proxy',
            'V0Candidates_mask',
            'V0Candidates_pt_proxy',
            'V0Candidates_px_proxy',
            'V0Candidates_py_proxy',
            'V0Candidates_pz_proxy',
            'V0Candidates_e_proxy',
            'JetsConstituents_dEdx_wires_value_masked',
            'JetsConstituents_PID_pval_wires_kaon_masked',
            'JetsConstituents_PID_pval_wires_pi_masked'
          ]
        }
        return names

    # add new variables to events dict

    # basic kinematics
    events['JetsConstituents_mask'] = ak.ones_like(events['JetsConstituents_pt'])
    events['JetsConstituents_pt_log'] = np.log(events['JetsConstituents_pt'])
    events['JetsConstituents_e_log'] = np.log(events['JetsConstituents_e'])
    events['JetsConstituents_drrel'] = np.hypot(events['JetsConstituents_thetarel'], events['JetsConstituents_phirel'])
    events['Jets_pt_log'] = np.log(events['Jets_pt'])
    events['Jets_e_log'] = np.log(events['Jets_e'])
    events['Jets_mass_log'] = np.log(events['Jets_mass'])

    # secondary vertices
    events['SecondaryVertices_mask'] = ( events['SecondaryVertices_chi2Normalized'] > 0. )
    events['SecondaryVertices_pt_proxy'] = np.multiply(events['SecondaryVertices_p'], np.sin(events['SecondaryVertices_thetarel']))
    events['SecondaryVertices_px_proxy'] = np.multiply(events['SecondaryVertices_pt_proxy'], np.cos(events['SecondaryVertices_phirel']))
    events['SecondaryVertices_py_proxy'] = np.multiply(events['SecondaryVertices_pt_proxy'], np.sin(events['SecondaryVertices_phirel']))
    events['SecondaryVertices_pz_proxy'] = np.multiply(events['SecondaryVertices_p'], np.cos(events['SecondaryVertices_thetarel']))
    events['SecondaryVertices_e_proxy'] = np.sqrt(np.square(events['SecondaryVertices_px_proxy']) + np.square(events['SecondaryVertices_py_proxy']) + np.square(events['SecondaryVertices_pz_proxy']) + np.square(events['SecondaryVertices_mass']))
    
    # V0 candidates
    events['V0Candidates_mask'] = ( events['V0Candidates_chi2Normalized'] > 0. )
    events['V0Candidates_pt_proxy'] = np.multiply(events['V0Candidates_p'], np.sin(events['V0Candidates_thetarel']))
    events['V0Candidates_px_proxy'] = np.multiply(events['V0Candidates_pt_proxy'], np.cos(events['V0Candidates_phirel']))
    events['V0Candidates_py_proxy'] = np.multiply(events['V0Candidates_pt_proxy'], np.sin(events['V0Candidates_phirel']))
    events['V0Candidates_pz_proxy'] = np.multiply(events['V0Candidates_p'], np.cos(events['V0Candidates_thetarel']))
    events['V0Candidates_e_proxy'] = np.sqrt(np.square(events['V0Candidates_px_proxy']) + np.square(events['V0Candidates_py_proxy']) + np.square(events['V0Candidates_pz_proxy']) + np.square(events['V0Candidates_mass']))

    # dEdx masking
    JetsConstituents_p = np.sqrt(np.square(events['JetsConstituents_pt']) + np.square(events['JetsConstituents_pz'])) # auxiliary variable
    JetsConstituents_dedx_mask = ((JetsConstituents_p>1) & (np.abs(events['JetsConstituents_charge'])>0) & (events['JetsConstituents_dEdx_wires_value']>0.8) & (events['JetsConstituents_isChargedHad']>0.5))
    events['JetsConstituents_dEdx_wires_value_masked'] = np.where(JetsConstituents_dedx_mask, events['JetsConstituents_dEdx_wires_value'], 0)
    events['JetsConstituents_PID_pval_wires_kaon_masked'] = np.where(JetsConstituents_dedx_mask, events['JetsConstituents_PID_pval_wires_kaon'], 0)
    events['JetsConstituents_PID_pval_wires_pi_masked'] = np.where(JetsConstituents_dedx_mask, events['JetsConstituents_PID_pval_wires_pi'], 0)
    return events


def preprocess_jets(jets, prepdict, translation=None):
    '''
    Do formatting and preprocessing of input variables.
    Input arguments:
      - jets: jets dict of the form {'<variable name>': awkward array, ...}
      - prepdict: a dictionary with preprocessing parameters, as returned by weaver.
      - translation: a dictionary with variable name translations.
    Returns:
      - data dict of the form {'<input name>': numpy array, ...}
    Note: need to keep in sync with how preprocessing is performed in weaver.
    See here: https://github.com/hqucms/weaver-core/blob/9b92955bc89ecada29f5d4e6b9a21f4f0a28f8c5/weaver/utils/dataset.py#L21
    '''
    data = {}
    # loop over input names
    for key in prepdict['input_names']:
        thisdata = []
        var_length = prepdict[key]['var_length']
        # loop over variables
        for varname in prepdict[key]['var_names']:
            params = prepdict[key]['var_infos'][varname]
            if translation is not None: varname = translation.get(varname, varname)
            values = jets[varname]
            # do preprocessing
            values = _clip((values - params['median']) * params['norm_factor'],
                           params['lower_bound'], params['upper_bound'])
            # do padding
            values = _pad(values, var_length, value=params['pad'])
            # check for nan
            if np.any(np.isnan(values)):
                values = np.nan_to_num(values)
            # check for inf
            if np.any(np.isinf(values)):
                values = np.where(np.isinf(values), params['replace_inf_value'], values)
            thisdata.append(values)
        thisdata = np.array(thisdata).astype('float32')
        thisdata = thisdata.transpose([1,0,2])
        data[key] = thisdata
    return data
        

def make_inference_session(modelname, num_threads=8):
    model = onnx.load(modelname)
    onnx.checker.check_model(model)

    session_options = onnxruntime.SessionOptions()
    session_options.inter_op_num_threads = num_threads
    session_options.intra_op_num_threads = num_threads
    return onnxruntime.InferenceSession(modelname, session_options)


def infer_jets(jets, modelname, prepdict, translation=None, batch_size=None, session=None):

    # get the data in correct format
    data = preprocess_jets(jets, prepdict, translation=translation)
    #if 'part' in modelname:
    if True: # now set to true by default since "part" is not always in model name...
        # (somehow some keys are missing in the onnx model inputs; not clear if this is expected,
        # or if the model will be evaluated correctly without it, but seems to be fine...)
        if 'points' in data.keys(): data.pop('points')
        if 'pf_points' in data.keys(): data.pop('pf_points')
        if 'sv_points' in data.keys(): data.pop('sv_points')
        if 'v0_points' in data.keys(): data.pop('v0_points')

    # divide in batches
    ndata = data[list(data.keys())[0]].shape[0]
    if batch_size is None or ndata <= batch_size: batches = [data]
    else:
        batches = []
        idx = 0
        while idx < ndata:
            batch = {key: values[idx:idx+batch_size] for key, values in data.items()}
            batches.append(batch)
            idx += batch_size
    print(f'[INFO in infer_jets]: received {ndata} instances for inference.')
    print(f'[INFO in infer_jets]: split instances into {len(batches)} batches of size {batch_size}')

    if session is None:
        session = make_inference_session(modelname)

    # run the inference
    outputs = []
    nclasses = len(prepdict['output_names'])
    for batch_idx, batch in enumerate(batches):
        print(f'[INFO in infer_jets]: running inference on batch {batch_idx+1} / {len(batches)}...', end='\r')
        batch_outputs = session.run(None, batch)[0]

        # do some sanity checks
        if batch_outputs.shape[1] != nclasses:
            msg = f'Expected {nclasses} output classes, but found array of shape {batch_outputs.shape}.'
            raise Exception(msg)
        for idx in range(batch_outputs.shape[1]):
            nanfrac = np.count_nonzero(np.isnan(batch_outputs[:,idx])) / batch_outputs.shape[0]
            if nanfrac > 0.1:
                msg = f'Found batch with > 10% NaNs, probably something is wrong.'
                raise Exception(msg)

        # get the scores
        scores = {}
        for idx, output_name in enumerate(prepdict['output_names']):
            scores[output_name] = batch_outputs[:,idx]
        outputs.append(scores)
    print()

    # concatenate batch outputs
    new_outputs = {}
    for output_name in prepdict['output_names']:
        new_outputs[output_name] = np.concatenate([output[output_name] for output in outputs])
    outputs = new_outputs

    # rename scores
    keys = list(outputs.keys())
    for key in keys:
        newname = 'score_' + key.split('_')[-1]
        outputs[newname] = outputs.pop(key)

    return outputs


def infer_events(events, modelname, prepdict, do_add_variables=False, **kwargs):
    # do preprocessing
    if do_add_variables: events = add_variables(events)
    # get variables per-jet and per-constituent
    jets_vars = [varname for varname in events.fields if varname.startswith('Jets_')]
    constituents_vars = [varname for varname in events.fields if varname.startswith('JetsConstituents_')]
    sv_vars = [varname for varname in events.fields if varname.startswith('SecondaryVertices_')]
    v0_vars = [varname for varname in events.fields if varname.startswith('V0Candidates_')]
    # check if at least one per-jet variable was provided (needed for flattening and un-flattening)
    if len(jets_vars)==0:
        msg = 'Need at least one per-jet variable in events to get the correct shape for flattening and un-flattening.'
        raise Exception(msg)
    # do flattening (needed for inference which is essentially per-jet level)
    jets_shape = ak.num(events[jets_vars[0]])
    jets = {}
    for varname in constituents_vars + jets_vars + sv_vars + v0_vars:
        jets[varname] = ak.flatten(events[varname], axis=1)
    # special handling of batches with no jets
    if len(jets[constituents_vars[0]])==0:
        outputs = {}
        for key in prepdict['output_names']: outputs[key] = np.array([])
    else:
        # run inference on jets
        outputs = infer_jets(jets, modelname, prepdict, **kwargs)
    # unflatten scores back into events shape
    outputs = {key: ak.unflatten(val, jets_shape) for key, val in outputs.items()}
    # make event score out of per-jet scores
    # update: store individual jet scores (in awkward array) for later more advanced combinations;
    #         do not make standard combinations anymore as they also depend on the jet selection
    #         (to be applied at a later stage)
    event_outputs = {}
    for key, val in outputs.items():
        output_name = 'Jets_score_' + key.split('_')[-1]
        event_outputs[output_name] = val
    return event_outputs
