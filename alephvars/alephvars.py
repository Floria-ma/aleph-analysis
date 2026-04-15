import os
import sys
import math
import numpy as np
import awkward as ak


def ipsig_prob(ipsig):
    '''
    Build probability scores for impact parameter significances.
    Input arguments:
    - ipsig: awkward array of impact parameter significances.
    Returns:
    - array of the same shape as input with probability scores.
    The probability distribution is constructed from the negative side of the ipsigs.
    '''

    # separate the negative and the positive ipsigs
    negipsig = ipsig[ipsig < 0]

    # define a probability distribution for the (absolute value of the) negative ipsigs
    dist = np.abs(ak.flatten(negipsig, axis=None).to_numpy())
    norm = len(dist)

    # approch 1: naive (very slow)
    #def integral(val):
    #    # note: very slow, to update implementation later!
    #    # note: also assumes val is a 2D awkward array
    #    res = []
    #    for l1 in val:
    #        res.append([])
    #        for el in l1:
    #            res[-1].append(np.sum(dist>np.abs(float(el))) / norm)
    #    res = ak.Array(res)
    #    return res
    #probs = integral(ipsig)
    
    # approach 2: same but with np.vectorize (still very slow)
    #def integral(val):
    #    return np.sum(dist>np.abs(float(val))) / norm
    #vectorized_integral = np.vectorize(integral)
    #ipsig_num = ak.num(ipsig)
    #probs = ak.unflatten(vectorized_integral(ak.flatten(ipsig)), ipsig_num)

    # approach 3: histogram instead of full array
    bins = np.concatenate((np.linspace(0, 5, num=1000), np.linspace(5, np.quantile(dist, 0.99)*1.5, num=500)))
    bincenters = (bins[:-1] + bins[1:])/2
    hist = np.histogram(dist, bins=bins)[0]
    def integral(val):
        return np.sum(hist[bincenters>=abs(float(val))])/norm
    vectorized_integral = np.vectorize(integral)
    ipsig_num = ak.num(ipsig)
    probs = ak.unflatten(vectorized_integral(ak.flatten(ipsig)), ipsig_num)

    factor = ak.where(ipsig<0, -ak.ones_like(ipsig), ak.ones_like(ipsig))
    probs = np.multiply(probs, factor)

    return probs


def jet_ipsig_prob(ipsig, prob=None):
    '''
    Combine per-track impact parameter probabilities into per-jet probabilities.
    '''

    # recalculate per-track probabilities if needed
    if prob is None: prob = ipsig_prob(ipsig)

    # select only positive probabilities
    prob = prob[prob>0]

    # make product
    p = ak.prod(prob, axis=1).to_numpy()

    # calculate terms for normalization factor
    # note: do this in fixed-length array style for speed;
    #       need to take sums only up to appropriate value later!
    n = ak.num(prob).to_numpy()
    maxn = np.amax(n)
    normfactors = np.ones((len(n), maxn))
    for j in range(1, maxn):
        normfactors[:, j] = np.power(-np.log(p), j)/math.factorial(j)

    # set superfluous values to zero before taking sum
    mask = ak.ones_like(prob)
    mask = ak.fill_none(ak.pad_none(mask, maxn, axis=1), 0).to_numpy()
    normfactors = np.multiply(normfactors, mask)
    normfactors[:, 0] = 1

    # take the sum to obtain the normalization factor
    normfactors = np.sum(normfactors, axis=1)

    # normalization
    pj = np.multiply(p, normfactors)

    # do clipping to range 0-1
    pj = np.clip(pj, a_min=0, a_max=1)
    
    return pj

def mass_ipsig_prob(ipsig, track_vectors, prob=None, threshold=1.8):
    '''
    Alternative method of combining per-track impact parameter probabilities
    into a per-jet probability, using the invariant mass.
    '''

    # recalculate per-track probabilities if needed
    if prob is None: prob = ipsig_prob(ipsig)

    # select only positive probabilities
    mask = (prob>0)
    prob = prob[mask]
    track_vectors = track_vectors[mask]

    # sort tracks inversely to their probability
    sorted_ids = ak.argsort(prob, ascending=True)
    prob = prob[sorted_ids]
    track_vectors = track_vectors[sorted_ids]

    # approach 1: naive (slow)
    '''
    def invmass(vectors):
        mass = (np.sqrt(np.square(np.sum(vectors.e))
                - np.square(np.sum(vectors.px))
                - np.square(np.sum(vectors.py))
                - np.square(np.sum(vectors.pz))
               ))
        return mass

    # loop over jets and combinations of tracks
    res = np.ones(len(track_vectors))
    for jetidx in range(len(track_vectors)):
        for trackidx in range(len(track_vectors[jetidx])):
            mass = invmass(track_vectors[jetidx, :trackidx+1])
            if mass > threshold:
                res[jetidx] = prob[jetidx, trackidx]
                break
    '''

    # approach 2
    def invmass(vectors, maxidx):
        mass = (np.sqrt(np.square(np.sum(vectors[:,:maxidx+1].e, axis=1))
                - np.square(np.sum(vectors[:,:maxidx+1].px, axis=1))
                - np.square(np.sum(vectors[:,:maxidx+1].py, axis=1))
                - np.square(np.sum(vectors[:,:maxidx+1].pz, axis=1))
               ))
        return mass
    masses = np.zeros((len(track_vectors), 5))
    for maxidx in range(5): masses[:, maxidx] = invmass(track_vectors, maxidx)
    ids = np.argmax(masses>threshold, axis=1)
    res = np.ones(len(prob))
    for jetidx in range(len(prob)):
        if len(prob[jetidx]) > 0 and ids[jetidx]>0:
            res[jetidx] = prob[jetidx, ids[jetidx]]

    return res

def unit(v):
    n = np.linalg.norm(v)
    if n == 0:
        return v
    return v/n

def p_vec(px, py, pz):
    return np.array([px, py, pz])

def thrust_value(axis, momentum):
    axis = unit(axis)
    num = sum(abs(np.dot(p, axis)) for p in momentum)
    den = sum(np.linalg.norm(p) for p in momentum)
    return 0.0 if den == 0 else num / den

def thrust_axis(momentum):
    """
    momentum: list/array of 3-vectors, shape (N, 3)
    returns: (best_axis, best_thrust)
    """
    best_axis = None
    best_thrust = -1.0

    # use particle directions as candidate axes
    for p in momentum:
        if np.linalg.norm(p) == 0:
            continue
        for axis in (unit(p), -unit(p)):
            t = thrust_value(axis, momentum)
            if t > best_thrust:
                best_thrust = t
                best_axis = axis

    return best_axis, best_thrust