"""Public-SDK driver: tinker.ServiceClient for model/fb calls, raw HTTP for
optim_step + durable-future polling (same path the 008/013 probes used).
"""

import os
import time


class GateStepError(RuntimeError):
    pass


class SDKDriver:
    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        step_timeout_s: float = 1800,
    ):
        self.base_url = base_url or os.environ.get(
            "TINKER_BASE_URL", "http://localhost:8000"
        )
        self.api_key = api_key or os.environ.get("TINKER_API_KEY", "tml-dev-key")
        os.environ.setdefault("TINKER_BASE_URL", self.base_url)
        os.environ.setdefault("TINKER_API_KEY", self.api_key)
        self.step_timeout_s = step_timeout_s
        self._sc = None

    @property
    def service_client(self):
        if self._sc is None:
            import tinker

            self._sc = tinker.ServiceClient()
        return self._sc

    def create_training_client(
        self,
        base_model: str,
        rank: int,
        seed: int | None = None,
    ):
        # Context length and debug mode are server configuration
        # (NEMORL_MAX_SEQ_LEN, NEMORL_DEBUG_TRAIN_ONLY); the client declares neither.
        kwargs = {"base_model": base_model, "rank": rank}
        if seed is not None:
            kwargs["seed"] = seed
        t0 = time.time()
        tc = self.service_client.create_lora_training_client(**kwargs)
        print(f"model_id: {tc.model_id} (boot {time.time() - t0:.1f}s)", flush=True)
        return tc

    def delete_model(self, model_id: str) -> bool:
        """Release the model's GPU resources. A gate MUST call this when it
        finishes: miles does not auto-reap models, and a leftover model (plus
        SGLang engines that outlive `ray stop`) wedges the next gate's
        create_model in placement. Never raises — cleanup must not mask a
        verdict."""
        import requests

        try:
            r = requests.post(
                f"{self.base_url}/api/v1/delete_model",
                headers={"X-API-Key": self.api_key, "Content-Type": "application/json"},
                json={"model_id": model_id},
                timeout=300,
            )
            if r.status_code == 200:
                print(f"deleted model {model_id}", flush=True)
                return True
            print(f"WARNING: delete_model {model_id} -> {r.status_code}: "
                  f"{r.text[:300]}", flush=True)
        except Exception as e:  # noqa: BLE001 - cleanup is best-effort
            print(f"WARNING: delete_model {model_id} failed: {e}", flush=True)
        return False

    def optim_step(self, model_id: str, lr: float = 0.0) -> dict:
        import requests

        hdrs = {"X-API-Key": self.api_key, "Content-Type": "application/json"}
        r = requests.post(
            f"{self.base_url}/api/v1/optim_step",
            headers=hdrs,
            json={"model_id": model_id, "adam_params": {"learning_rate": lr}},
        )
        r.raise_for_status()
        rid = r.json()["request_id"]
        deadline = time.time() + self.step_timeout_s
        while time.time() < deadline:
            fr = requests.post(
                f"{self.base_url}/api/v1/retrieve_future/{rid}", headers=hdrs
            )
            if fr.status_code == 200:
                return fr.json()
            if fr.status_code == 408:
                time.sleep(2)
                continue
            raise GateStepError(f"optim_step failed ({fr.status_code}): {fr.text[:1500]}")
        raise GateStepError(f"optim_step timeout after {self.step_timeout_s}s")
