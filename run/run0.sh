export CUDA_VISIBLE_DEVICES='0'

uv run longitudinal.py \
    -cn longi.yaml \
    training.num_workers=12 \
    training.dev_num_workers=10 \
    training.prefetch_factor=5 \
    training.dev_prefetch_factor=5 \
    training.batch_size=4 \
    training.sampler=stratified \
    training.minority_samples_per_batch=1 \
    training.epochs=15 \
    wandb.dry_run=False \
    training.to_checkpoint=True \
    log.checkpoint_at_epoch=1 \
    optimizer.init_lr=8e-8 \
    optimizer.peak_lr=5e-6 \
    optimizer.end_lr=5e-7 \
    optimizer.warmup_epochs=5 \
    training.freeze_encoder=False \
    longitudinal.model=transformer \
    longitudinal.rnn_cell=lstm \
    longitudinal.rnn_hidden_dim=768 \
    longitudinal.blocks=8 \
    longitudinal.heads=12 \
    longitudinal.bidirectional=False \
    log.mae_use_checkpoint=pretty-thunder-81
