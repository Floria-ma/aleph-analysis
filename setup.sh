#!/usr/bin/bash

# Run as follows:
# source setup.sh

source /cvmfs/sw.hsf.org/key4hep/setup.sh

# Optional local overlay for packages that are not shipped with Key4hep.
# Create it with:
#   python -m venv ~/venvs/aleph-key4hep --system-site-packages
#   source ~/venvs/aleph-key4hep/bin/activate
#   python -m pip install iminuit
aleph_key4hep_venv="${ALEPH_KEY4HEP_VENV:-${HOME}/venvs/aleph-key4hep}"
if [ -f "${aleph_key4hep_venv}/bin/activate" ]; then
    source "${aleph_key4hep_venv}/bin/activate"
fi
unset aleph_key4hep_venv
