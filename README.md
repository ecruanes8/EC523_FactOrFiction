# EC523_FactOrFiction

This project aims to detect fake news using the LIAR dataset by comparing a BiLSTM baseline with fine-tuned BERT and RoBERTa models. We evaluate performance on both the original six-class classification task and a binary classification.

## Installation
All notebooks were developed and run on Google Colab with a T4 GPU runtime, so no local installation is required.

To run the notebooks yourself:

1. Open the notebook in [Google Colab](https://colab.research.google.com/)
2. Go to Runtime → Change runtime type → **T4 GPU**
3. Run the first cell to install dependencies if needed:

```python
!pip install torch transformers scikit-learn pandas numpy matplotlib tensorflow keras seaborn
```
## Usage

1. Download `train.tsv`, `valid.tsv`, and `test.tsv` from the LIAR dataset]
2. Upload them to Google Drive
3. Update the file path in the data loading cell to match your Drive location:
```python
train = pd.read_csv("/content/drive/MyDrive/YOUR_FOLDER/train.tsv", sep="\t", header=None)
```
4. Run all cells in order

The best model checkpoint is saved automatically as `best_model.pt` and reloaded for final test evaluation.

## Dataset
| Label | Description |
|---|---|
| `true` | Fully accurate |
| `mostly-true` | Mostly accurate with minor omissions |
| `half-true` | Partially accurate |
| `barely-true` | Some truth but misleading |
| `false` | Inaccurate |
| `pants-fire` | completely false |

Each example includes 13 metadeta fields: statement, subject, speaker, job, state, party, and cumulative truth counts. 
For binary classification, labels are collapsed into fake: pants-fire, false, and barely-true and true: half-true, mostly-true, and true.

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

### 4. RoBERTa (Binary)
- Tokenized with `roberta-base` or `roberta-large` tokenizer (max length = 128)
- `RobertaForSequenceClassification`, 2 output labels, dropout = 0.2
- Differential learning rates: backbone lr = 1×10⁻⁵, classifier head lr = 1×10⁻³
- Linear warmup (10% steps) + early stopping (patience = 3) on validation loss
- Best checkpoint saved and reloaded for test evaluation

## Results

| Model | Task | Test Accuracy |
|---|---|---|
| BiLSTM | 6-class | 22.89% |
| BERT | 6-class | 27.84% |
| RoBERTa-base | 6-class | 43.8% |
| RoBERTa-base | Binary | 71.6% |
| RoBERTa-large | Binary | 74.73% |

---

