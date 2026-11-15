# FLAIR Group 4: Multimodal Mood and Theme Tagging of Music

Multi-label mood/theme tagging on the [MTG-Jamendo dataset](https://github.com/MTG/mtg-jamendo-dataset)
(mood/theme subset, official split-0, 56 tags). The project tests whether adding short track text
(title, artist, album) to pre-trained audio features improves the rarest tags, using **late fusion**:
an audio model and a text model are trained separately and their tag probabilities are averaged.

## Method

* **Audio branch:** each track's audio (`audio-low` MP3, mono, 16 kHz) goes through the pre-trained
  Essentia MusiCNN model `msd-musicnn-1`. The 200-dimensional output of the penultimate layer
  (`model/dense/BiasAdd`) is averaged over all 3-second patches, giving one 200-d embedding per track.
  The embeddings are standardised and fed to one-vs-rest logistic regression (L2, C = 1.0, liblinear).
* **Text branch:** title + artist + album as one string, TF-IDF (unigrams and bigrams, up to 10,000
  features, fitted on training text only), then the same one-vs-rest logistic regression.
* **Late fusion:** `P_fusion = (P_audio + P_text) / 2` for every track and tag.
* **Uncommon tags:** the 14 tags (bottom 25%) with the fewest training examples.
* **Metrics:** PR-AUC-macro (primary), PR-AUC-micro, ROC-AUC-macro, and PR-AUC on the uncommon and other
  tags, with a paired bootstrap (1,000 resamples) for the uncommon-tag gain of fusion over audio.

## Results

Validation (3,802 tracks):

| | PR-AUC-macro | PR-AUC-micro | ROC-AUC-macro | PR-AUC uncommon (14) | PR-AUC other (42) |
|---|---|---|---|---|---|
| Audio-only | 0.0961 | 0.1026 | 0.7190 | 0.0877 | 0.0990 |
| Text-only | 0.1163 | 0.1210 | 0.6281 | 0.0991 | 0.1220 |
| Late fusion | 0.1317 | 0.1392 | 0.7434 | 0.1259 | 0.1336 |

Test (4,231 tracks, evaluated once after all choices were fixed):

| | PR-AUC-macro | PR-AUC-micro | ROC-AUC-macro | PR-AUC uncommon (14) | PR-AUC other (42) |
|---|---|---|---|---|---|
| Audio-only | 0.1242 | 0.1467 | 0.7401 | 0.0912 | 0.1352 |
| Text-only | 0.1248 | 0.1361 | 0.6402 | 0.0607 | 0.1461 |
| Late fusion | 0.1514 | 0.1805 | 0.7586 | 0.1020 | 0.1678 |

Uncommon-tag gain of late fusion over audio-only: 0.0382 on validation (95% CI 0.0282 to 0.0504) and
0.0108 on test (95% CI 0.0057 to 0.0193). The executed notebook shows the full per-tag results and the
success-criteria checks.

## Files

| File | What it is |
|---|---|
| `FLAIR_Group_4_notebook.ipynb` | Main notebook, fully executed: data, text and audio branches, late fusion, validation and test results. |
| `FLAIR_Group_4_results/` | Result tables (`validation_results.csv`, per-tag CSVs, LaTeX table rows), `run_info.json` with package versions, and `musicnn_embeddings_all.npz` (all 17,982 track embeddings: `ids`, `emb`). |
| `FLAIR_Group_4_mood_tagging.ipynb` | Earlier text-only notebook, kept for reference. |

## Running it locally

`essentia-tensorflow` has no Windows build, so use Linux or WSL2 (tested on Ubuntu 24.04, Python 3.12).

```bash
python3 -m venv ~/flair-venv && source ~/flair-venv/bin/activate
pip install scikit-learn==1.6.1 essentia-tensorflow==2.1b6.dev1389 numpy pandas scipy joblib jupyter nbconvert ipykernel
git clone https://github.com/Lerbernard/flair-mood-tagging.git && cd flair-mood-tagging
```

The notebook clones MTG-Jamendo at commit `cafd8e2` itself. Two switches in the setup cell control the run:

* `RUN_AUDIO`: `True` downloads the audio (46 GB in 100 archives, processed and deleted one at a time)
  and extracts the MusiCNN embeddings; this takes many hours. `False` reuses the saved embeddings.
  To reuse them, put `FLAIR_Group_4_results/musicnn_embeddings_all.npz` into
  `FLAIR_Group_4_results/musicnn_embeddings/`.
* `RUN_FINAL_TEST`: `True` evaluates once on the held-out test split. Only turn it on after every
  model choice is final.

Then execute the whole notebook:

```bash
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 \
    '--ExecutePreprocessor.extra_arguments=["--IPKernelApp.capture_fd_output=False"]' \
    FLAIR_Group_4_notebook.ipynb
```

The `capture_fd_output=False` kernel option is needed when `RUN_AUDIO = True`: without it, Essentia's
log output can fill Jupyter's output pipe and stall the kernel. An NVIDIA GPU speeds up the embedding
step; install NVIDIA's CUDA 11 / cuDNN 8 pip packages (`nvidia-*-cu11`) and add their `lib` folders to
`LD_LIBRARY_PATH`.

## Data

The dataset audio and the MusiCNN model are downloaded at run time and are not stored in this repository.
See the [MTG-Jamendo license](https://github.com/MTG/mtg-jamendo-dataset#license) for terms of use.
