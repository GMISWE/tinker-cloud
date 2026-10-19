# tinkercloud tests

The directory is the suite and the file name is the module under test.
Every test directory is a package, so same-named files can live in two suites.

| Suite | Needs | Runs where |
|---|---|---|
| `unit/` | nothing beyond the server's own imports (CPU torch for a few) | CI and the pod; bare `pytest` |
| `conformance/` | the `tinker` SDK; spawns a real server on the `fake` backend | the pod (the SDK repo is internal); bare `pytest` |
| `gpu/` | 3+ GPUs, the backend's training env; starts its own Ray | the pod, by path; every test carries the `gpu` marker |

`unit/` mirrors `training/`: `core/`, `models/`, `services/`, `storage/`,
`checkpoints/`, `proto/`, `backends/<name>/`, plus `gates/` for the gate suite's
pure layers. A test that imports `tinkercloud.training.x.y` lives at
`unit/x/test_y.py`.

`conformance/` drives the HTTP API through the real SDK (auth, model info,
loss registry, sampler routing, sampling params, checkpoints, sessions,
per-model ordering, the proto wire). It runs against `--backend fake` today;
the same assertions are the contract every backend must meet.

## Running

The tests import the server as the `tinkercloud` package: the checkout must be
reachable under that name (a symlink named `tinkercloud` pointing at this
directory, with its parent on `PYTHONPATH`). On the cluster:

```bash
scripts/pod_pytest.sh                                   # unit + conformance on the default pod
scripts/pod_pytest.sh tinkercloud-automodel tests/unit/backends/automodel -q
KX_TIMEOUT=1700 scripts/pod_pytest.sh tinkercloud-automodel tests/gpu -v   # server and Ray head stopped first
```

The script pushes the local working tree (sha-verified) to a scratch dir on
the pod and runs pytest there; the deployed `/app` is untouched. GPU suites
need the pod's server and Ray head stopped: they start their own Ray.

`unit/proto/test_wire_sdk.py` (the codec driven by the SDK's own converters) is
skipped unless a `tinker >= 0.25` is importable. To run the conformance suite
under an upstream SDK without touching the pod's install, put one in a scratch
target and prepend it to `PYTHONPATH` for the test process only:

```bash
kubectl exec tinkercloud-nemorl -- pip install --target /tmp/pytest-$USER/sdk027 "tinker==0.27.0"
kubectl exec tinkercloud-nemorl -- bash -c "cd /tmp/pytest-$USER/app && \
  PYTHONPATH=/tmp/pytest-$USER/sdk027:/tmp/pytest-$USER python -m pytest tests/conformance -q"
```

Tests that only skip on the pod say why in their skip reason (`-rs` lists
them): NeMo RL's converter and buffering tests need `nemo_rl` installed; the
gate replay tests need the monorepo's recorded results beside this checkout.
