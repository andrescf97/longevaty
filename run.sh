export CUDA_VISIBLE_DEVICES='0'

uv run longitudinal.py \
    -cn longi.yaml \
    attention.use_fusion_layer=True \
    attention.use_attention=True \
    attention.use_cls=True \
    attention.use_mean_token=True \
    training.freeze_encoder=True \
    training.freeze_mha=True \
    training.batch_size=4 \
    wandb.dry_run=False \
    training.to_checkpoint=True \
    training.num_workers=12 \
    training.dev_num_workers=10 \
    training.prefetch_factor=5 \
    training.dev_prefetch_factor=5 \
    training.sampler=weighted \
    training.minority_samples_per_batch=1 \
    training.epochs=15 \
    log.checkpoint_at_epoch=1 \
    optimizer.init_lr=1e-9 \
    optimizer.peak_lr=1e-7 \
    optimizer.end_lr=1e-9 \
    optimizer.warmup_epochs=1 \
    longitudinal.model=transformer \
    longitudinal.blocks=6 \
    longitudinal.heads=6 \
    # training.freeze_encoder=True \
    # training.freeze_mha=True \
    # longitudinal.bidirectional=False \
    # longitudinal.rnn_cell=lstm \
    # longitudinal.rnn_hidden_dim=768 \