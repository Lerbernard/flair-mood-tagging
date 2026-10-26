# FLAIR Group 4: Music Mood/Theme Tagging

Multi-label mood and theme tagging on the [MTG-Jamendo dataset](https://github.com/MTG/mtg-jamendo-dataset),
built in Google Colab.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Lerbernard/flair-mood-tagging/blob/main/FLAIR_Group_4_mood_tagging.ipynb)

## What the notebook does

1. Clones the MTG-Jamendo dataset and loads the `autotagging_moodtheme.tsv` annotations (18,486 tracks, 59 tags).
2. Uses the official `split-0` train / validation / test splits (9,949 / 3,802 / 4,231 tracks) and the 56 tags they contain.
3. Joins track, artist and album names from `raw.meta.tsv`, recovering 35 rows that pandas misparses.
4. Builds a text-only baseline: TF-IDF (unigrams and bigrams) over track, artist and album names, with a one-vs-rest logistic regression.
5. Reports validation PR-AUC (macro AP 0.1163, micro AP 0.121) and per-tag average precision.
6. Saves the model, vectorizer, predictions and per-tag results, and sets up Essentia's MusiCNN (`msd-musicnn-1`) for the audio model.

## Running it

Open the notebook in Colab with the badge above and run the cells in order. It installs
`essentia-tensorflow` and downloads the MusiCNN model itself. The last cells use
`google.colab.files`, so they only work inside Colab.

## Data

The dataset and the MusiCNN model are downloaded at run time and are not stored in this repository.
See the [MTG-Jamendo license](https://github.com/MTG/mtg-jamendo-dataset#license) for terms of use.
