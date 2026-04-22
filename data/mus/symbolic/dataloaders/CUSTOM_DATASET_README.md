# Custom Music Dataset Loader

## Overview

The generic `CustomMusicDataset` loader supports any custom tokenized music dataset following a standard directory structure. This allows you to train on custom datasets like `jordan-progrock-dataset` or any other dataset you prepare.

## Expected Directory Structure

```
/home/rebcecca/orcd/pool/music_datasets/
├── jordan-progrock-dataset/
│   ├── midi/                          (optional, for reference)
│   ├── quarantined_midis/             (optional, for corrupt files)
│   └── tokens/
│       ├── anticipation/
│       │   ├── tokenized-events-train.txt
│       │   ├── tokenized-events-validation.txt
│       │   └── tokenized-events-test.txt
│       ├── anticipation-vanilla/      (alternative tokenizer)
│       │   └── (same structure)
│       └── miditok/                   (alternative tokenizer)
│           └── (same structure)
└── giga-midi/
    ├── midi/
    └── tokens/
        ├── anticipation/
        │   └── (same structure)
        └── anticipation-vanilla/
            └── (same structure)
```

## Token File Format

Each line in the token files contains **space-separated token IDs** representing a single tokenized music sequence:

```
123 456 789 234 567 890 ...
98 76 54 32 11 99 ...
...
```

## Usage

### 1. Training with a Custom Dataset

To train on your custom music dataset, use the following command:

```bash
python train_model.py \
    --dataset_name "jordan-progrock-dataset" \
    --dataset_music_tokenizer_type "anticipation-vanilla" \
    --modality "MUS_SYMB" \
    --model_name "ebt" \
    --context_length 512 \
    ... (other args)
```

### 2. Command Line Arguments

- `--dataset_name`: Name of the dataset folder (e.g., `jordan-progrock-dataset`)
- `--dataset_music_tokenizer_type`: Tokenizer subdirectory name (default: `anticipation`)
  - Options: `anticipation`, `anticipation-vanilla`, `miditok`, or any other subdirectory in the `tokens/` folder

### 3. Example Training Scripts

See `job_scripts/mus/pretrain/example_custom_dataset.sh` for a complete example.

## Implementation Details

### CustomMusicDataset Class

Located in `data/mus/symbolic/dataloaders/custom_music_dataloader.py`

Key features:
- Automatically detects dataset location from `CUSTOM_STORAGE_PATH` environment variable (defaults to `/home/rebcecca/orcd/pool/music_datasets`)
- Loads all sequences from the token file into memory
- Pads or truncates sequences to `context_length` during batching
- Returns samples as `{"input_ids": tensor}` for compatibility with the model

### How It Works

1. **Dataset Discovery**: When you specify `--dataset_name "jordan-progrock-dataset"`:
   - Looks for `/home/rebcecca/orcd/pool/music_datasets/jordan-progrock-dataset/tokens/`
   - Checks for the specified tokenizer subdirectory (e.g., `anticipation-vanilla`)

2. **Sequence Loading**: 
   - Reads `tokenized-events-{split}.txt` files (train, validation, test)
   - Each line is parsed as space-separated integers
   - Each sequence is stored as a PyTorch tensor

3. **Batching**:
   - Sequences are padded to `context_length` with zeros (if too short)
   - Sequences are truncated to `context_length` (if too long)

## Adding a New Custom Dataset

To add support for a new custom dataset:

1. Create the directory structure:
   ```bash
   mkdir -p /home/rebcecca/orcd/pool/music_datasets/my-dataset/tokens/anticipation
   ```

2. Add your tokenized sequences to token files:
   ```bash
   # Each line: space-separated token IDs
   echo "123 456 789 234 567" >> tokenized-events-train.txt
   ```

3. Train using the dataset:
   ```bash
   python train_model.py \
       --dataset_name "my-dataset" \
       --dataset_music_tokenizer_type "anticipation" \
       ...
   ```

## Troubleshooting

### "Token file not found"

Error message shows the expected path. Ensure:
- Dataset folder exists at `/home/rebcecca/orcd/pool/music_datasets/{dataset_name}`
- `tokens/` subdirectory exists
- Tokenizer type subdirectory exists (e.g., `anticipation-vanilla`)
- Token files are named `tokenized-events-{split}.txt`

### "No sequences found"

This means the token file exists but is empty or contains only empty lines. Check that:
- Token files contain space-separated integers on each line
- No extra whitespace or invalid characters

### Setting Custom Data Path

If your datasets are stored elsewhere, set the environment variable:
```bash
export CUSTOM_STORAGE_PATH="/path/to/music_datasets"
python train_model.py --dataset_name "my-dataset" ...
```

## Environment Variables

- `CUSTOM_STORAGE_PATH`: Base directory containing all custom datasets (default: `/home/rebcecca/orcd/pool/music_datasets`)
