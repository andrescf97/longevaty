import os
import torch 

def load_checkpointed_state(loc, ckpt_name, device, model, optimizer, scheduler, scaler):
    os.makedirs(loc, exist_ok=True)

    loc = os.path.join(loc, ckpt_name)
    if not os.path.exists(loc):
        return 0
    checkpoint = torch.load(loc, map_location=device)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    if scheduler is not None:
        scheduler.load_state_dict(checkpoint['scheduler'])
    scaler.load_state_dict(checkpoint['scaler'])
    return checkpoint['epochs'] + 1

def save_checkpoint(loc, run_name, ckpt_name, model, epoch, optimizer, scheduler, scaler):
    if not os.path.isdir(os.path.join(loc, run_name)):
        os.makedirs(os.path.join(loc, run_name), exist_ok=True)
    loc = os.path.join(loc, run_name, ckpt_name)
    checkpoint = {
        "epochs": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
    }

    with open(loc, "wb") as fp:
        torch.save(checkpoint, fp)
