import os
import sys
import math
import numpy as np
import awkward as ak

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

    events = ak.with_field(events, ak.Array(thrust_vals), "Event_thrust")
    events = ak.with_field(events, ak.Array(thrust_x), "Event_thrust_x")
    events = ak.with_field(events, ak.Array(thrust_y), "Event_thrust_y")
    events = ak.with_field(events, ak.Array(thrust_z), "Event_thrust_z")
    events = ak.with_field(events, ak.Array(costhrust), "Event_costhrust")
    return events

def theta_difference(events):
    thrust_costheta = np.abs(events["Event_thrust_z"])
    jet_costheta = np.abs(np.cos(events["Jets_theta"]))

    diff = np.abs(thrust_costheta - jet_costheta)

    events = ak.with_field(events, diff, "thrust_jets_costheta_diff")

    return events