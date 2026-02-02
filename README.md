## Feature Extraction

The backbone of our longitudinal models relies on feature embeddings extracted from the fine-tuned **Lungevity** vision transformer. To achieve this, we utilize a custom model wrapper located at:

`adlm_lft/longitudinal/delta/feature_extractor.py`

This wrapper executes the model's `forward()` method with **frozen weights**, allowing us to bypass the final classification head and capture rich, intermediate feature representations (e.g., CLS tokens, attention pooling, max pooling) directly from the LDCT scans.

### How to Run

To extract features for your dataset, run the following script:

```bash
python adlm_lft/feature_extractor.py

```

#### Configuration Requirements

Before running the script, you must update the configuration to match your specific experiment. We recommend using `adlm_lft/configs/feature_extraction.yaml` as a template. Ensure the following parameters are set correctly:

* **Checkpoint Path:** Point to the correct `.pt` checkpoint file for the fine-tuned Lungevity model.
* **Dataset Path:** Provide paths to the MONAI Dataset `.json` files. These must be pre-split into `train`, `dev`, and `test`.
* **Patch Size:** Update the patch size to match the architecture of the loaded checkpoint (e.g., `[8, 8, 8]` or `[16, 16, 16]`).
* **Output Directory:** Specify the destination folder where the extracted features will be saved.
* **File Suffix:** Define a descriptive suffix for the output files. The script will save files using the format `[split]_[file_suffix].pt` (e.g., `train_patch8.pt`).

### Output Format

The script generates a `.pt` file containing a serialized dictionary. The structure of this dictionary is:

```python
{
    exam_id: feature_tensor
}

```

* **Key (`exam_id`):** The unique string identifier for the specific scan/exam. This corresponds directly to the unique `"exam"` key found in the MONAI dataset metadata.
* **Value (`feature_tensor`):** The extracted embedding tensor for that scan (concatenated vector of dimension 2376).

This format allows for efficient querying (`O(1)` lookup) of feature tensors during the training of downstream longitudinal models.