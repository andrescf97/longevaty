import orbax.checkpoint as ocp
from flax import nnx


def load_checkpoint(mngr):
    try:
        state = mngr.restore(mngr.latest_step())
        start_epoch = mngr.latest_step()
    except FileNotFoundError as e:
        start_epoch = 0
        state = None

    return start_epoch, state