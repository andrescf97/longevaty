import orbax.checkpoint as ocp
from flax import nnx
import os
import torch 


def load_checkpoint(mngr):
    try:
        state = mngr.restore(mngr.latest_step())
        start_epoch = mngr.latest_step() + 1
    except FileNotFoundError as e:
        start_epoch = 0
        state = None

    return start_epoch, state


def load_checkpointed_state(loc, ckpt_name, device, model, optimizer, scheduler, scaler, new_learning_rate):
    loc = loc + ckpt_name
    if not os.path.exists(loc):
        return 0
    else:
        print("Resuming from checkpoint")
    checkpoint = torch.load(loc, map_location=device)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    scheduler.load_state_dict(checkpoint['scheduler'])
    scaler.load_state_dict(checkpoint['scaler'])
    
    # Change the learning rate if a new one is provided
    if new_learning_rate is not None:
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_learning_rate
    
    return checkpoint['epochs'] + 1

def save_checkpoint(loc, model, epoch, optimizer, scheduler, scaler):
    checkpoint = {
        "epochs": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict()
    }

    with open(loc, "wb") as fp:
        torch.save(checkpoint, fp)