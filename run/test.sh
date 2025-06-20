export CUDA_VISIBLE_DEVICES='3'

uv run test_longi.py \
    -cn longi.yaml \
    training.num_workers=32 \
    training.dev_num_workers=14 \
    training.prefetch_factor=10 \
    training.dev_prefetch_factor=16 \
    training.batch_size=1 \
    training.minority_samples_per_batch=2 \
    training.epochs=10 \
    wandb.dry_run=True \
    training.to_checkpoint=False \
    longitudinal.model=transformer \
    longitudinal.rnn_cell=lstm \
    longitudinal.rnn_hidden_dim=768 \
    longitudinal.blocks=4 \
    longitudinal.heads=6 \
    longitudinal.bidirectional=False \
    log.use_checkpoint=blooming-snow-142 \
    log.ckpt_load=vit

