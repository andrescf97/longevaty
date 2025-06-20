export CUDA_VISIBLE_DEVICES='3'

uv run mae.py \
    -cn mae-all.yaml \
    training.batch_size=8 \
    training.num_workers=20 \
    training.dev_num_workers=12 \
    training.prefetch_factor=6 \
    training.dev_prefetch_factor=6 \
    log.log_at_these_steps=5000 \
    log.use_checkpoint=test \
    training.to_checkpoint=True \
    training.epochs=800 \
    training.mask_ratio=0.75 \
    log.log_scans_at_these_epochs=5 \
    optimizer.warmup_epochs=20