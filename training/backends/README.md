# Backends

One directory per backend; the directory list is the list of backends. Each
implements the `TrainingBackend` ABC in `base.py` and registers in
`factory.py`. A backend owns its `builder.py` (request → launch config),
`converter.py` (wire datums → native tensors), `config.py` and the engine
integration; nothing backend-specific lives outside its directory.

## Wire fields and what each converter makes of them

Tinker sends `Datum` objects: `model_input` tokens plus `loss_fn_inputs`
as `TensorData`. The per-token fields and their native names:

| Wire (SDK)               | Miles (`List[Tensor]` per sample) | NeMo RL (`BatchedDataDict` [B, S]) | Automodel (per-request batch) |
|--------------------------|-----------------------------------|------------------------------------|-------------------------------|
| `tokens`                 | `tokens`                          | `input_ids`                        | `input_ids`                   |
| `loss_masks` / `weights` | `loss_masks` + `weights`          | `token_mask`                       | `weights`                     |
| `advantages`             | `advantages`                      | `advantages`                       | `advantages`                  |
| `logprobs`               | `log_probs`                       | `prev_logprobs`                    | `logprobs`                    |
| `target_tokens`          | appended as the final target      | shifted labels                     | `target_tokens`               |
| (computed)               | padding to the DP width           | `input_lengths` [B], `sample_mask` [B] | row sums, rank shards     |

The unit tests under `tests/unit/backends/<name>/test_converter.py` are the
executable form of this table.

## Optimizer state files (automodel)

A training checkpoint root holds the interchange adapter (`hf_adapter/`) and the
tenant's optimizer state under `optimizer/`, one file named for the family
(`backends/automodel/adapter.py`):

| family | file | keys per LoRA parameter |
|---|---|---|
| `adamw` | `optimizer/adamw.safetensors` | `<peft key>.exp_avg`, `.exp_avg_sq`, `.step` |
| `dimuon` | `optimizer/dimuon.safetensors` | `<peft key>.momentum` |

The family is read from the file name. A load with `optimizer=true` across
families is refused at the API with a 400 naming both families (the checkpoint
record carries the family); a mismatching file that reaches the worker is a
`StateMismatch`. A weights-only load crosses families, which is how a run
switches optimizer.
Before the first step the file exists and is empty. Each backend declares the
families it serves in `SUPPORTED_OPTIMIZERS`; TinkerCloud's `dimuon` is a
stand-in for Thinking Machines' unpublished algorithm (`dion.Muon` applied to
each LoRA factor, weight decay 0, dion's `spectral_norm` learning-rate scaling).
