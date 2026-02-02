import torch

def longitudinal_collate_fn(batch):
    """
    Collate a list of patient samples into a batch for the DataLoader.
    Each patient has a variable number of visits (T), so we pad sequences
    to max_T (the longest sequence in this batch).

    Returns a dict containing:
      - cls_seq:      [B, max_T, D]
      - timepoints:   [B, max_T]
      - y:            [B, max_T]
      - y_seq:        [B, max_T, 6]
      - y_mask:       [B, max_T, 6]
      - padding_mask: [B, max_T]
      - pid:          list length B
      - cancer_laterality: list length B
    """

    B = len(batch)
    D = batch[0]["cls_seq"].shape[1]   

    # compute max_T from the actual patients in this batch
    max_T = max(item["cls_seq"].shape[0] for item in batch)

    # Allocate padded batch tensors
    cls_batch = torch.zeros(B, max_T, D)
    time_batch = torch.zeros(B, max_T)
    y_batch = torch.zeros(B, max_T)
    time_at_event_batch = torch.zeros(B, max_T)
    y_seq_batch = torch.zeros(B, max_T, 6)
    y_mask_batch = torch.zeros(B, max_T, 6)

    # True = padded position, False = real visit
    padding_mask = torch.ones(B, max_T, dtype=torch.bool)

    pids = []
    cancer_laterality = []
    prefix_lens = []


    for i, item in enumerate(batch):
        T = item["cls_seq"].shape[0]   # number of visits for this patient

        # Fill the first T slots with real data
        cls_batch[i, :T] = item["cls_seq"]
        time_batch[i, :T] = item["timepoints"]
        y_batch[i, :T] = item["y"]
        time_at_event_batch[i, :T] = item["time_at_event"]
        y_seq_batch[i, :T] = item["y_seq"]
        y_mask_batch[i, :T] = item["y_mask"]

        # Mark real visits as NOT padding
        padding_mask[i, :T] = False
        prefix_lens.append(item.get("prefix_len", None))

        pids.append(item["pid"])
        cancer_laterality.append(item["cancer_laterality"])

    return {
        "pid": pids,
        "cls_seq": cls_batch,             
        "timepoints": time_batch,         
        "y": y_batch,                
        "time_at_event": time_at_event_batch,     
        "y_seq": y_seq_batch,             
        "y_mask": y_mask_batch,          
        "padding_mask": padding_mask,    
        "cancer_laterality": cancer_laterality,
        "prefix_len": prefix_lens,

    }
