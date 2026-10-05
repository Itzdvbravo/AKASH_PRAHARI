# Train DynamicEarthNet semantic change detection in Google Colab

This repository includes [`notebooks/train_dynamicearthnet_semantic_cd_colab.ipynb`](../notebooks/train_dynamicearthnet_semantic_cd_colab.ipynb). It trains the 49-class ordered land-cover transition model on Colab's GPU, evaluates on AOIs held out from the training split, and saves the best checkpoint and JSON reports to Google Drive.

## Run it

1. Push the code version containing the semantic training and evaluation scripts to the configured GitHub repository. In the notebook, set `REPO_REF` to the appropriate branch or commit if it is not `main`.
2. Copy `data/dynamicearthnet/dynamicearthnet-video-71psnr.tacozip` (about 2.2 GB) from the training computer to `MyDrive/TerraEyes/data/dynamicearthnet/dynamicearthnet-video-71psnr.tacozip`. The notebook cannot read files on the training computer directly.
3. Open the notebook in Google Colab, choose a GPU runtime, mount Drive, and run the cells in order. The first run clones the project and decodes/samples imagery, which takes time. The decoded cache is kept on Colab's local disk for faster reads while connected.
4. If the runtime disconnects during training, rerun the setup and data preparation cells, then rerun the training cell. It resumes from the best checkpoint stored on Drive. Data sampling is repeated because it is held in Colab memory during a run.
5. Review `training_outputs/dynamicearthnet_semantic_training.json` and `training_outputs/dynamicearthnet_semantic_evaluation.json` in Drive. Download `training_outputs/mamba_dynamicearthnet_semantic.pt` to the local model path shown in the notebook.

The training script holds sampled image pairs in memory. The default sample size uses 8 monthly pairs and 8 tiles per pair for each training and validation AOI; adjust the notebook arguments if Colab RAM is insufficient. Evaluation processes all 16 tiles across the selected validation AOIs and two configured date pairs, so it can take substantially longer than training.

Validation AOIs are sampled from the dataset's indexed training AOIs; this is not a claim of official test-set performance. Keep `semantic_mamba` opt-in until the held-out binary and semantic metrics have been reviewed. The checkpoint path can be set with `TERRAEYES_SEMANTIC_MAMBA_CHECKPOINT_PATH`.
