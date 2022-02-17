docker build -t mpnn .
docker run \
    -v '/home/and/projects/hacks/molecule-activity/models:/app/models' \
    -v '/home/and/projects/hacks/molecule-activity/scripts:/app/scripts' \
    -v '/home/and/projects/hacks/molecule-activity/logs:/app/logs' \
    --gpus all \
    --log-opt max-size=10m \
    --log-opt max-file=1 \
    --name mpnn \
    --rm \
    mpnn python scripts/mpnn/train_mpnn.py
