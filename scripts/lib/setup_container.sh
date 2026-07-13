#!/usr/bin/env bash
# Shared in-container setup for TinkerCloud dev environments.
# Runs INSIDE the k8s pod or docker container after /tmp/code.tar.gz is in place.
# Single source of truth so k8s and docker deployments cannot drift.
#
# PROFILE (env, default nemo_rl) cases the base-image-specific steps:
#   nemo_rl  overlay RL/nemo_rl, editable-install SDK+cookbook + extra deps.
#   miles    miles pre-installed in the image; editable-install SDK+cookbook + deps.
#   bionemo  deploy code + scripts/evo2 only; NO nemo_rl overlay, NO SDK/cookbook
#            installs (the bionemo image's deps are tightly pinned — don't perturb).
#   megatron_bridge  build the bionemo-recipes evo2_megatron recipe env
#            (megatron-bridge) so the megatron_bridge backend can import it.
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

if [ "$PROFILE" = megatron_bridge ]; then
  # cu13 recipe env: build the bionemo-recipes evo2_megatron recipe (megatron-bridge)
  # so backends/megatron_bridge/ can import megatron.bridge + evo2_classifier.
  R=/workspace/evo2_megatron
  if [ ! -d "$R/.venv" ]; then
    rm -rf "$R"; mkdir -p /workspace
    mv /tmp/bionemo-recipes/recipes/evo2_megatron "$R"
    # patch the torch-2.12 weights_only=True load bug in the savanna converter
    sed -i 's/weights_only=True, mmap=True/weights_only=False, mmap=True/' \
      "$R/src/bionemo/evo2/utils/checkpoint/savanna_to_mbridge.py" || true
    echo "building recipe env (megatron-bridge; ~15 min) ..."
    ( cd "$R" && bash .ci_build.sh ) > /data/recipe_build.log 2>&1
  else
    echo "recipe env already built ($R/.venv)"
  fi
  "$R/.venv/bin/python" -c 'import importlib.util as u; assert u.find_spec("megatron.bridge"); print("megatron.bridge OK")'
  test -f /app/training/backends/megatron_bridge/backend.py && echo "megatron_bridge backend OK"
  echo "NOTE: server not started — backend is stubs. Recipe at $R; venv $R/.venv."
  echo SETUP_DONE
  exit 0
fi

# nemo_rl overlays its worker onto the image's install (picked up at next
# create_model); miles + its stack are already baked into the miles image.
if [ "$PROFILE" = nemo_rl ]; then
  PKG=$(python3 -c 'import nemo_rl, os; print(os.path.dirname(nemo_rl.__file__))')
  cp -r /tmp/RL/nemo_rl/. "$PKG/"
elif [ "$PROFILE" = miles ]; then
  # Overlay the GMI miles fork onto the image's editable install: the base image's
  # baked miles predates the Tinker Ray-orchestration interface (forward_backward_only,
  # apply_optimizer_step(learning_rate=), apply_optimizer_step_and_sync). Fork main is
  # the maintained branch (PR merges fix_hackathon_rebased -> main).
  MILES_REPO="${MILES_REPO:-https://github.com/GavinZhu-GMI/miles.git}"
  MILES_REF="${MILES_REF:-main}"
  MPKG=$(python3 -c 'import miles, os; print(os.path.dirname(miles.__file__))')
  rm -rf /tmp/miles_src
  git clone -q --depth 1 --branch "$MILES_REF" "$MILES_REPO" /tmp/miles_src
  cp -r /tmp/miles_src/miles/. "$MPKG/"
  python3 -c 'import miles; print("miles fork overlaid OK ('"$MILES_REF"')")'
fi

# SDK + cookbook editable installs
mv /tmp/tinker_gmi /work/tinker_gmi
mv /tmp/tinker-cookbook /work/tinker-cookbook
pip install -e /work/tinker_gmi --no-deps -q
# dev-mode bundles may lack .git -> setuptools-scm needs a pretend version
SETUPTOOLS_SCM_PRETEND_VERSION=${SETUPTOOLS_SCM_PRETEND_VERSION:-0.1.0} \
  pip install -e /work/tinker-cookbook --no-deps -q

# deps missing from the base image that --no-deps skips
# (distro: SDK; chz/termcolor/blobfile/tiktoken: cookbook;
#  sympy/pylatexenc/math-verify: math_rl recipe extras)
pip install -q distro chz termcolor blobfile tiktoken cloudpickle rich anyio \
  'httpx[http2]' sympy pylatexenc math-verify

RUNTIME=$([ "$PROFILE" = miles ] && echo miles || echo nemo_rl)
python3 -c "import tinker, tinker_cookbook, $RUNTIME; print('imports OK')"
PYTHONPATH=/app python3 -c 'import training; print("training OK")'
echo SETUP_DONE
