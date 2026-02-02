# Instructions to run tinker-cookbook recipe math-rl

First pull image from dockerhub
```
docker pull gmicloudai/tinkercloud:dev-local
```
Then run the tinkercloud service with:
```
docker run -d --name tinkercloud-rlve --gpus all \
      -v /data:/data \
      -v ${YOUR_TINKER_COOKBOOK_PATH}:/workspace/tinker-cookbook \
      -v ${YOUR_TINKER_PATH}:/workspace/tinker_gmi \
      --network host \
      --shm-size=64g \
      -e ALLOW_PARTIAL_BATCHES=true \
      -e TINKER_API_KEY=slime-dev-key \
      gmicloudai/tinkercloud:dev-local
```
Now, execute the following command:

```
docker exec -e TINKER_API_KEY=slime-dev-key -e WANDB_MODE=disabled tinkercloud-rlve bash -c \
      "cd /workspace/tinker-cookbook && python -m tinker_cookbook.recipes.math_rl.train \
        model_name=/data/models/Llama-3.2-1B \
        renderer_name=role_colon \
        base_url=http://localhost:8000 \
        group_size=4 \
        groups_per_batch=100 \
        learning_rate=1e-6 \
        n_batches=5 \
        log_path=/data/logs/math-rl-lr1e6 \
        behavior_if_log_dir_exists=delete \
        advantage_estimator=center"
```
