#!/usr/bin/env bash
# Shared in-container setup for TinkerCloud dev environments.
# Runs INSIDE the k8s pod or docker container after /tmp/code.tar.gz is in place.
# Single source of truth so k8s and docker deployments cannot drift.
#
# PROFILE (env, default nemo_rl) cases the base-image-specific steps:
#   nemo_rl  overlay RL/nemo_rl, editable-install SDK+cookbook + extra deps.
#   bionemo  deploy code + scripts/evo2 only; NO nemo_rl overlay, NO SDK/cookbook
#            installs (the bionemo image's deps are tightly pinned — don't perturb).
set -euo pipefail
PROFILE="${PROFILE:-nemo_rl}"

cd /tmp && tar xzf code.tar.gz
mkdir -p /app /work /data/metadata /data/trajectories /data/.cache/huggingface

# TinkerCloud server -> /app (run via PYTHONPATH=/app python3 -m training)
rm -rf /app/training /app/tests /app/scripts /work/tinker_gmi /work/tinker-cookbook
mv /tmp/tinker-cloud/training /app/training
mv /tmp/tinker-cloud/tests /app/tests
[ -d /tmp/tinker-cloud/scripts/evo2 ] && { mkdir -p /app/scripts; mv /tmp/tinker-cloud/scripts/evo2 /app/scripts/evo2; }

if [ "$PROFILE" = bionemo ]; then
  # Faithful Evo2 env: code + evo2 helpers in place; bionemo.evo2 provides the
  # heavy stack. Skip nemo_rl overlay + SDK/cookbook (pinned-env safety).
  # NB: `import training` pulls fastapi (absent here) — verify files, not import.
  python3 -c 'import bionemo.evo2; print("bionemo.evo2 OK")'
  command -v evo2_convert_to_nemo2 >/dev/null && echo "evo2 entrypoints OK"
  test -f /app/training/backends/megatron_bridge/backend.py && echo "megatron_bridge backend OK"
  echo SETUP_DONE
  exit 0
fi

# NeMo RL worker overlay onto the image's install (picked up at next create_model)
PKG=$(python3 -c 'import nemo_rl, os; print(os.path.dirname(nemo_rl.__file__))')
cp -r /tmp/RL/nemo_rl/. "$PKG/"

# SDK + cookbook editable installs
mv /tmp/tinker_gmi /work/tinker_gmi
mv /tmp/tinker-cookbook /work/tinker-cookbook
pip install -e /work/tinker_gmi --no-deps -q
# dev-mode bundles may lack .git -> setuptools-scm needs a pretend version
SETUPTOOLS_SCM_PRETEND_VERSION=${SETUPTOOLS_SCM_PRETEND_VERSION:-0.1.0} \
  pip install -e /work/tinker-cookbook --no-deps -q

# deps missing from the nemo-rl image that --no-deps skips
# (distro: SDK; chz/termcolor/blobfile/tiktoken: cookbook;
#  sympy/pylatexenc/math-verify: math_rl recipe extras)
pip install -q distro chz termcolor blobfile tiktoken cloudpickle rich anyio \
  'httpx[http2]' sympy pylatexenc math-verify

python3 -c 'import tinker, tinker_cookbook, nemo_rl; print("imports OK")'
PYTHONPATH=/app python3 -c 'import training; print("training OK")'
echo SETUP_DONE
