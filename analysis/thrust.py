import os
import sys
import math
import numpy as np
import awkward as ak
'''
def unit(v):
    n = np.linalg.norm(v)
    if n == 0:
        return v
    return v/n

def p_vec(px, py, pz):
    return np.array([px, py, pz])

# for each particle's direction, compute the thrust value as abs(all particles' momentum dot that direction) / (sum of all particles' momentum magnitudes)
def thrust_value(axis, momentum):
    axis = unit(axis)
    num = sum(abs(np.dot(p, axis)) for p in momentum)
    den = sum(np.linalg.norm(p) for p in momentum)
    return 0.0 if den == 0 else num / den

def thrust_axis(momentum):
    momentum = np.asarray(momentum, dtype=float)

    if len(momentum) == 0:
        return np.zeros(3), 0.0

    best_axis = np.zeros(3)
    best_thrust = 0.0
    found_nonzero = False

    for p in momentum:
        if np.linalg.norm(p) == 0:
            continue

        found_nonzero = True
        axis = unit(p)
        t = thrust_value(axis, momentum)

        if t > best_thrust:
            best_thrust = t
            best_axis = axis

    if not found_nonzero:
        return np.zeros(3), 0.0

    return best_axis, best_thrust

def add_thrust_variables(events):
    thrust_vals = []
    thrust_x = []
    thrust_y = []
    thrust_z = []
    costhrust = []
    thrust_phi = []

    for i in range(len(events)):
        px_nested = events["JetsConstituents_px"][i]
        py_nested = events["JetsConstituents_py"][i]
        pz_nested = events["JetsConstituents_pz"][i]

        px = ak.to_numpy(ak.flatten(px_nested, axis=None))
        py = ak.to_numpy(ak.flatten(py_nested, axis=None))
        pz = ak.to_numpy(ak.flatten(pz_nested, axis=None))

        momenta = np.stack([px, py, pz], axis=1) if len(px) > 0 else np.zeros((0, 3))
        axis, thrust = thrust_axis(momenta)

        thrust_vals.append(thrust)
        thrust_x.append(axis[0])
        thrust_y.append(axis[1])
        thrust_z.append(axis[2])
        costhrust.append(abs(axis[2]))
        thrust_phi.append(np.arctan2(axis[1], axis[0]))

    events = ak.with_field(events, ak.Array(thrust_vals), "Event_thrust")
    events = ak.with_field(events, ak.Array(thrust_x), "Event_thrust_x")
    events = ak.with_field(events, ak.Array(thrust_y), "Event_thrust_y")
    events = ak.with_field(events, ak.Array(thrust_z), "Event_thrust_z")
    events = ak.with_field(events, ak.Array(costhrust), "Event_costhrust")
    events = ak.with_field(events, ak.Array(thrust_phi), "Event_thrust_phi")
    return events

def theta_difference(events):
    thrust_costheta = np.abs(events["Event_thrust_z"])
    jet_costheta = np.abs(np.cos(events["Jets_theta"]))

    diff = np.abs(thrust_costheta - jet_costheta)

    events = ak.with_field(events, diff, "thrust_jets_costheta_diff")

    return events
'''
def _seed_axes(px, py, pz, pmag):
    """
    Per-event seed axis: the single-particle direction that maximizes the
    thrust numerator sum_i |p_i . n|. Vectorized over all events.

    px, py, pz, pmag are jagged (events x particles) awkward arrays.
    Returns (nx, ny, nz) regular numpy arrays, one axis per event.
    """
    # Candidate axes are the unit particle directions; for candidate c the
    # numerator is sum_i |p_i . p_c| / |p_c|. Build the |events x cand x part|
    # dot-product magnitude and pick, per event, the candidate with the max sum.
    # Done per event via awkward to avoid densifying the full 3D tensor.
    # dot_ic = px_i*px_c + py_i*py_c + pz_i*pz_c
    # We compute, for each candidate c, S_c = sum_i |dot_ic| / |p_c|, then argmax.
    counts = ak.num(px)
    n_events = len(px)

    nx = np.zeros(n_events)
    ny = np.zeros(n_events)
    nz = np.zeros(n_events)

    # Densify per event only over its own particles (small), but loop-free across
    # events by grouping equal multiplicities. In practice constituent counts
    # cluster, so this is a handful of vectorized batches rather than N events.
    offsets = np.asarray(ak.to_numpy(counts))
    order = np.argsort(offsets, kind="stable")
    px_np = ak.to_numpy(ak.flatten(px)) if ak.sum(counts) > 0 else np.zeros(0)
    py_np = ak.to_numpy(ak.flatten(py)) if ak.sum(counts) > 0 else np.zeros(0)
    pz_np = ak.to_numpy(ak.flatten(pz)) if ak.sum(counts) > 0 else np.zeros(0)
    pmag_np = ak.to_numpy(ak.flatten(pmag)) if ak.sum(counts) > 0 else np.zeros(0)
    starts = np.concatenate([[0], np.cumsum(offsets)])

    # group events by identical multiplicity -> one dense batched einsum each
    for m in np.unique(offsets):
        if m == 0:
            continue
        ev_idx = np.where(offsets == m)[0]
        # gather this batch's particle blocks into (batch, m, 3)
        rows = np.empty((len(ev_idx), m, 3))
        mags = np.empty((len(ev_idx), m))
        for r, e in enumerate(ev_idx):
            s = starts[e]
            rows[r, :, 0] = px_np[s:s + m]
            rows[r, :, 1] = py_np[s:s + m]
            rows[r, :, 2] = pz_np[s:s + m]
            mags[r] = pmag_np[s:s + m]
        # candidate directions = rows / |p| (guard zero-momentum particles)
        safe = np.where(mags > 0, mags, 1.0)
        cand = rows / safe[:, :, None]
        # dot[b, i, c] = p_i . cand_c ; numerator S[b, c] = sum_i |dot| (den |p_c|=1)
        dot = np.einsum("bic,bjc->bij", rows, cand)  # (batch, i, c)
        S = np.sum(np.abs(dot), axis=1)              # (batch, c)
        S = np.where(mags > 0, S, -np.inf)           # never pick a zero-mom seed
        best = np.argmax(S, axis=1)                  # (batch,)
        chosen = cand[np.arange(len(ev_idx)), best]  # (batch, 3)
        nx[ev_idx] = chosen[:, 0]
        ny[ev_idx] = chosen[:, 1]
        nz[ev_idx] = chosen[:, 2]

    return nx, ny, nz


def _refine_axes(px, py, pz, nx, ny, nz, n_iter=4):
    """
    Fixed-point thrust maximization, vectorized over all events:
        n <- normalize( sum_i sign(p_i . n) p_i )
    Repeated n_iter times. Jagged sums via ak.sum over the particle axis.
    """
    for _ in range(n_iter):
        dot = px * nx + py * ny + pz * nz          # jagged (events x particles)
        s = np.sign(dot)
        new_x = ak.to_numpy(ak.sum(s * px, axis=1))
        new_y = ak.to_numpy(ak.sum(s * py, axis=1))
        new_z = ak.to_numpy(ak.sum(s * pz, axis=1))
        norm = np.sqrt(new_x ** 2 + new_y ** 2 + new_z ** 2)
        norm = np.where(norm > 0, norm, 1.0)
        nx, ny, nz = new_x / norm, new_y / norm, new_z / norm
    return nx, ny, nz


def add_thrust_variables(events, n_iter=5):
    """
    Compute the event thrust axis and value from jet constituents and attach:
    Event_thrust, Event_thrust_{x,y,z}, Event_costhrust, Event_thrust_phi.

    Fully vectorized across events; n_iter controls the refinement passes
    (4 is ample for convergence in practice; set 0 to use only the
    single-particle seed, matching the original approximation).
    """
    # Flatten each event's jet-constituent momenta (events x particles).
    px = ak.flatten(events["JetsConstituents_px"], axis=2)
    py = ak.flatten(events["JetsConstituents_py"], axis=2)
    pz = ak.flatten(events["JetsConstituents_pz"], axis=2)
    pmag = np.sqrt(px ** 2 + py ** 2 + pz ** 2)

    nx, ny, nz = _seed_axes(px, py, pz, pmag)
    if n_iter > 0:
        nx, ny, nz = _refine_axes(px, py, pz, nx, ny, nz, n_iter=n_iter)

    # Thrust value with the final axis: sum_i |p_i . n| / sum_i |p_i|.
    dot = px * nx + py * ny + pz * nz
    num = ak.to_numpy(ak.sum(np.abs(dot), axis=1))
    den = ak.to_numpy(ak.sum(pmag, axis=1))
    thrust = np.where(den > 0, num / np.where(den > 0, den, 1.0), 0.0)

    # Events with no particles / zero axis -> zeroed axis, thrust 0.
    axis_norm = np.sqrt(nx ** 2 + ny ** 2 + nz ** 2)
    empty = (den <= 0) | (axis_norm == 0)
    nx = np.where(empty, 0.0, nx)
    ny = np.where(empty, 0.0, ny)
    nz = np.where(empty, 0.0, nz)
    thrust = np.where(empty, 0.0, thrust)

    events = ak.with_field(events, ak.Array(thrust), "Event_thrust")
    events = ak.with_field(events, ak.Array(nx), "Event_thrust_x")
    events = ak.with_field(events, ak.Array(ny), "Event_thrust_y")
    events = ak.with_field(events, ak.Array(nz), "Event_thrust_z")
    events = ak.with_field(events, ak.Array(np.abs(nz)), "Event_costhrust")
    events = ak.with_field(events, ak.Array(np.arctan2(ny, nx)), "Event_thrust_phi")
    return events


def theta_difference(events):
    thrust_costheta = np.abs(events["Event_thrust_z"])
    jet_costheta = np.abs(np.cos(events["Jets_theta"]))
    diff = np.abs(thrust_costheta - jet_costheta)
    events = ak.with_field(events, diff, "thrust_jets_costheta_diff")
    return events