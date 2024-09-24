python dinov2/run/train/train.py \
    --nodes 1 \
    --config-file dinov2/configs/train/mineral_vitb14.yaml\
    --output-dir output \
    --ngpus 1 \
    --no-resume \
    train.dataset_path=MineralDataset:root=/mnt/e/MineralDataset/dataset:split=train
