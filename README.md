# EC523_FactOrFiction

This project aims to detect fake news using the LIAR dataset, comparing BiLSTM baseline against a fine-tuned BERT model model to classify on 6 classes. 

| Label | Description |
|---|---|
| `true` | Fully accurate |
| `mostly-true` | Mostly accurate with minor omissions |
| `half-true` | Partially accurate |
| `barely-true` | Some truth but misleading |
| `false` | Inaccurate |
| `pants-fire` | completely false |

## Database Split:
| Split | Size |
|---|---|
| Train | 10,240 |
| Validation | 1,284 |
| Test | 1,267 |

## Pipeline

### 1. Data Preprocessing

- Loaded train, validation, and test splits from the LIAR .tsv files
- Kept only the label and statement columns
- Applied text cleaning:
  - Lowercased all text
  - Removed punctuation, URLs, and hashtags
  - Expanded contractions (e.g. `didn't` → `did not`)
- Encoded labels numerically using `LabelEncoder`

### 2. BiLSTM Baseline

- Tokenized text using Keras `Tokenizer`
- Padded sequences to uniform length
- Model architecture:
  - Embedding layer (128-dim)
  - Bidirectional LSTM (64 units)
  - Dense layer (64 units, ReLU)
  - Output layer (6 units, Softmax)
- Trained for 5 epochs with `sparse_categorical_crossentropy` loss

### 3. BERT

- Tokenized text using `bert-base-uncased` tokenizer (max length: 128)
- Used `BertForSequenceClassification` with 6 output labels
- Optimizer: AdamW (learning_rate =2e-5) with linear warmup scheduler
- Trained for 5 epochs on GPU

## Results

| Model | Test Accuracy |
|---|---|
| BiLSTM | 22.89% |
| BERT | 27.84% |
