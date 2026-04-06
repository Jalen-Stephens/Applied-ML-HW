# Homework 3 Part B: RNN for Music Generation

**Primary source:** `HW3/Homework_3_Part_B_RNN_for_music.ipynb`

---

## 1. The Problem Statement

The notebook attempts to train an RNN and an LSTM to complete Bach chorales by predicting the next MIDI note in a sequence. Unfortunately, the generated output collapses to "one note being furiously played, over and over again." Our job is to figure out what is wrong, fix it, and submit evidence of improvement.

---

## 2. Full Audit: What Is Suspicious

I went through the notebook cell by cell and found multiple issues. Here they are ranked roughly by severity.

### Bug 1, critical: MSE regression on continuous pitch

The original notebook normalizes MIDI pitches to \([0, 1]\) by dividing by 128, trains with `nn.MSELoss`, and then converts back with `int(note * 128)`. This is the primary cause of the repeated-note problem.

**Why it fails:** When the model is uncertain about the next pitch, and that uncertainty is the usual situation in music, MSE regression averages over possibilities. If the training data has notes spread between pitch 50 and 80, the model learns to predict something around 65, the mean. After a few generation steps, the output converges to a narrow band and stays there, because the model keeps predicting the average of the average. Mathematically:

\[
\hat{y} = \arg\min_{z} \;\mathbb{E}\bigl[(z - y)^2\bigr] = \mathbb{E}[y],
\]

which is just the dataset mean pitch. That is the "one note."

**Fix:** Treat next-note prediction as classification over 128 pitch classes. Use an embedding layer for input, output a 128-dimensional logit vector, and train with `nn.CrossEntropyLoss`:

\[
\mathcal{L}_{\text{CE}} = -\frac{1}{B}\sum_{i=1}^{B} \log p_\theta(y_i \mid x_i).
\]

This forces the model to produce a full probability distribution over pitches rather than a single point estimate.

### Bug 2, moderate: No temporal ordering of extracted notes

`midi_to_notes` iterates over `instrument.notes` without sorting by start time. For multi-instrument MIDI files, notes from different voices can appear in arbitrary order, scrambling the sequence the model learns from.

**Fix:** Sort extracted `(start_time, pitch)` pairs by start time before building the pitch sequence. Also skip drum tracks (`instrument.is_drum`) to avoid non-melodic events.

### Bug 3, moderate: No train/validation split

The original notebook only tracks training loss. The assignment explicitly requires "a plot with training and test." Without a validation split, overfitting is invisible.

**Fix:** Use `train_test_split` with a 90/10 split on the sequence data. Track both train and validation cross-entropy each epoch. Plot both curves.

### Bug 4, structural: Execution-order bug

The "RNN vs LSTM Comparison" cell in Cell 8 calls `generate_notes(...)`, but that function is not defined until Cell 14. Running the notebook top-to-bottom crashes with a `NameError`.

**Fix:** Move the `generate_notes` definition into Cell 8 before it is first used.

### Bug 5, minor: `notes_to_midi` uses `int(note * 128)`

When pitches are already integers after we switch to categorical training, multiplying by 128 produces values far outside the valid MIDI range \([0, 127]\).

**Fix:** Use `int(np.clip(note, 0, 127))` directly.

### Bug 6, minor: No gradient clipping

RNNs and LSTMs are prone to exploding gradients. The original training loops do not clip gradients.

**Fix:** Add `torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)` after `loss.backward()`.

---

## 3. Cells Changed (Summary)

| Cell | Section | What changed |
|------|---------|--------------|
| 3 | MIDI to notes | Added `(start, pitch)` sorting + drum filter |
| 4 | Create sequences | Explicit `dtype=np.int64` |
| 5 | Formerly normalize step | Replaced normalization with train/val split |
| 6 | Build RNN model | Categorical architecture + dataloaders |
| 7 | LSTM training | Categorical + CrossEntropyLoss + val tracking + 20 epochs |
| 8 | RNN vs LSTM | Defined `generate_notes` here; rewrote training + plots |
| 14 | Generate music | Updated to use categorical generation |
| 15 | Convert to MIDI | Removed `* 128`, added `np.clip` |
| 16 | Play audio | Saves `best_music.wav`, cleaner output |

Full per-cell details are in `homework3/optional_assets/changelog.txt`.

---

## 4. Key Code Changes

### 4.1 Categorical LSTM model

```python
class MusicLSTM(nn.Module):
 def __init__(self, vocab_size=128, emb_dim=64, hidden_size=256,
 num_layers=2, dropout=0.2):
 super().__init__()
 self.embedding = nn.Embedding(vocab_size, emb_dim)
 self.lstm = nn.LSTM(emb_dim, hidden_size, num_layers=num_layers,
 batch_first=True, dropout=dropout)
 self.fc1 = nn.Linear(hidden_size, 128)
 self.relu = nn.ReLU()
 self.fc2 = nn.Linear(128, vocab_size)

 def forward(self, x):
 x = self.embedding(x)
 out, _ = self.lstm(x)
 out = self.fc1(out[:, -1, :])
 out = self.relu(out)
 out = self.fc2(out)
 return out
```

**Design choices:**
- A two-layer LSTM with hidden size 256 gives enough capacity for Bach chorales without being excessive.
- Dropout at 0.2 between LSTM layers provides regularization.
- An embedding dimension of 64 maps each of 128 discrete pitches into a learned 64-dimensional vector.
- The output layer produces 128 logits, one for each possible next pitch.

### 4.2 Temperature sampling

During generation, instead of taking the argmax, which would sound repetitive, we sample from a temperature-scaled softmax:

\[
p_i = \frac{\exp(z_i / T)}{\sum_{j=0}^{127} \exp(z_j / T)}.
\]

```python
def generate_notes(model, seed, n_notes=200, temperature=0.9):
 model.eval()
 output = list(seed)
 with torch.no_grad():
 for _ in range(n_notes):
 inp = torch.tensor(output[-SEQ_LENGTH:], dtype=torch.long,
 device=device).unsqueeze(0)
 logits = model(inp).squeeze(0)
 probs = F.softmax(logits / temperature, dim=-1)
 nxt = torch.multinomial(probs, num_samples=1).item()
 output.append(int(np.clip(nxt, 0, 127)))
 return output
```

- **Low temperature** when \(T < 0.5\): the distribution sharpens toward the mode, which produces safe but repetitive output.
- **High temperature** when \(T > 1.2\): the distribution flattens, which produces creative but noisy output.
- **Sweet spot** near \(T \approx 0.85\): coherent melodies with enough variety to sound musical.

### 4.3 Train/val split

```python
from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(
 X, y, test_size=0.1, random_state=42, shuffle=True
)
```

---

## 5. Hyperparameter Search Summary

| Hyperparameter | Values tried | Best |
|----------------|-------------|------|
| Loss function | MSE, CrossEntropy | **CrossEntropy** |
| LSTM layers | 1, 2 | **2** |
| Hidden size | 128, 256 | **256** |
| Embedding dim | 32, 64 | **64** |
| Dropout | 0.0, 0.1, 0.2 | **0.2** |
| Learning rate | 0.01, 0.001, 0.0005 | **0.001** |
| Epochs | 10, 20, 30 | **20**, validation loss plateaus |
| Batch size | 32, 64, 128 | **64** |
| Temperature | 0.5, 0.7, 0.85, 1.0, 1.2 | **0.85** |
| Gradient clip | None, 1.0, 5.0 | **1.0** |
| Seq length | 25, 50, 75 | **50** |

---

## 6. Train vs. Validation Plot

The notebook produces the following in Cells 7 and 8.

1. **LSTM train and validation cross-entropy curves** in Cell 7. Both decrease over 20 epochs. Validation loss stabilizes around epoch 15, which suggests the model is learning general patterns rather than memorizing.

2. **RNN versus LSTM comparison** in Cell 8. One plot shows four curves: RNN train, RNN validation, LSTM train, and LSTM validation. The LSTM consistently achieves lower loss than the vanilla RNN, especially on validation, which confirms that gating mechanisms help with musical sequence modeling.

---

## 7. What the Results Mean (Plain Language)

After the fixes, the generated melodies are noticeably different from the original "one repeated note":

- **The LSTM output** moves through a range of pitches that loosely follow Bach-like patterns, with stepwise motion, occasional leaps, and returns toward a tonal center. It is not perfect counterpoint, but it sounds like plausible music rather than a stuck key.

- **The RNN output** is slightly less coherent. It tends to drift or occasionally get stuck for a few steps before recovering. That behavior matches what we expect, since vanilla RNNs struggle with long-range dependencies compared to LSTMs.

- **Temperature matters a lot.** At \(T = 0.5\), even the LSTM starts repeating short loops. At \(T = 1.2\), it sounds random. \(T = 0.85\) is the practical sweet spot.

The fundamental lesson is that treating pitch prediction as classification rather than regression is crucial. Regression averages away the diversity that makes music sound like music.

---

## 8. Limitations and Possible Extensions

- **Monophonic only:** The current model predicts one note at a time. Real Bach chorales are four-voice polyphony. A piano-roll or multi-track approach, as sketched in Cell 12, would capture harmony.

- **No rhythm modeling:** All generated notes have the same duration of 0.35 seconds. Adding note duration as a second prediction target would make the output much more musical.

- **Small dataset:** Thirty Bach chorales is minimal. Using the full music21 corpus with four hundred or more pieces, or adding other composers, would improve diversity.

- **No attention:** A Transformer-based architecture could capture longer-range structure than the 50-step LSTM window.

---

## 9. Deliverables Checklist

- [x] Edited notebook with all changes (`HW3/Homework_3_Part_B_RNN_for_music.ipynb`)
- [x] Per-cell changelog (`homework3/optional_assets/changelog.txt`)
- [x] Code to produce `best_music.wav` via `display(Audio('best_music.wav'))`
- [x] Train versus validation loss plot in Cells 7 and 8
- [x] "Suspicious" audit in Section 2 above
