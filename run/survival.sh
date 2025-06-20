export CUDA_VISIBLE_DEVICES='2'

uv run survival.py \
    -cn survival.yaml \
    training.batch_size=4 \
    training.num_workers=10 \
    training.dev_num_workers=8 \
    training.prefetch_factor=5 \
    training.dev_prefetch_factor=5 \
    log.log_at_these_steps=2000 \
    log.mae_use_checkpoint=neat-bird-196 \
    training.epochs=50 \
    wandb.dry_run=False \
    loss.sw=1 \
    loss.aw=10 \
    optimizer.peak_lr=1e-5 \
    optimizer.init_lr=8e-7 \
    optimizer.end_lr=1e-6 \
    attention.heads=12 \
    attention.use_cls=True \
    attention.use_mean_token=True \
    transform.train_tf.MaskPatchesd_our.use_annotations=True \
    optimizer.warmup_epochs=5 \
    log.checkpoint_at_epoch=1 \
    training.to_checkpoint=True \
    model.fusion_layer=True \
    training.sampler=stratified \
    training.minority_samples_per_batch=1