# BERT Riksdagen classifier


This is a general repo for fine tuning KBBERT for specific tasks related to the SWERIK Corpora.

## NB Change of directory structure

The repo started as one task (note-seg) classification and the original scripts at project root and under `preprocess/` have been moved under `tasks/note-seg`. Tuning data has been moved from `data/` to `data/note-seg/`. Hard coded file paths still need to be updated. Similarly tests have not yet been run on join-segments, split-segments, or titles, which were imported from other repos -- check for hard-coded paths.

## What's here

### `tasks/`
Each task gets a subdirectory with it's own scripts.

#### `note-seg/`

Note-segment classification.

### `data/`

Each `tasks/` subdirectory has a corresponding `data/` subdirectory where fine tuning data is stored. 

### `models/`

Tuned models are stored here, but not committed to the GH repo (only attached to each release).

### General purpose scripts

Eventually the plan is to create some generic scripts at project root that will train and run a variety of classification models on the SWERIK Corpora with only a fine tuning data set and eventual minimal task-specific preprocessing.