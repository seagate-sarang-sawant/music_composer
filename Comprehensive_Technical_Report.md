# Comprehensive Technical Report: Deep Learning Approaches for Composer Classification

## Abstract

This technical report presents a comparative analysis of two distinct deep learning approaches for automated composer classification using MIDI data. The study examines two implementations: a traditional music21-based approach (Group 2 Model) and an advanced MidiTok-based approach (Sarang's Alternate Model). Both models employ hybrid CNN-LSTM architectures but differ significantly in their feature extraction methodologies, data preprocessing techniques, and model complexities. The analysis reveals critical insights into overfitting challenges, performance variations, and potential enhancement strategies for music information retrieval systems.

## 1. Introduction

### 1.1 Background

Music composer identification represents a challenging task in music information retrieval (MIR), requiring sophisticated pattern recognition capabilities to distinguish between compositional styles. The ubiquity of digital music formats and the growing need for automated music analysis have driven research toward deep learning solutions that can capture the nuanced characteristics of different composers' works.

### 1.2 Objective

The primary objective of this comparative study is to evaluate two distinct deep learning methodologies for composer classification:
- Traditional symbolic music representation using music21 library
- Advanced tokenization approach using MidiTok framework

The analysis focuses on methodology comparison, performance evaluation, and identification of enhancement opportunities to address overfitting and improve generalization capabilities.

## 2. Literature Review and Theoretical Framework

### 2.1 Music Information Retrieval

Music information retrieval encompasses computational approaches to analyze, understand, and organize musical data. In the context of composer classification, the challenge lies in extracting meaningful features that capture compositional style while maintaining computational efficiency.

### 2.2 Deep Learning in Music Analysis

Recent advances in deep learning have demonstrated significant potential in music analysis tasks. Convolutional Neural Networks (CNNs) excel at capturing local patterns in musical sequences, while Long Short-Term Memory (LSTM) networks effectively model temporal dependencies inherent in musical compositions.

### 2.3 Tokenization Approaches

Traditional approaches rely on symbolic representations (note names, MIDI numbers), while modern tokenization methods like MidiTok provide more sophisticated encoding schemes that capture musical semantics more effectively.

## 3. Methodology

### 3.1 Dataset Description

Both implementations utilize the Composer Dataset containing MIDI files from classical composers:
- **Composers**: Bach, Bartok, Byrd, Chopin, Handel, Hummel, Mendelssohn, Mozart, Schumann
- **Data Split**: Train/Development/Test partitions
- **File Format**: MIDI (.mid, .midi)
- **Class Distribution**: Approximately balanced across composers

### 3.2 Group 2 Model (Traditional Approach)

#### 3.2.1 Data Preprocessing
```python
# Feature Extraction using music21
def extract_note_sequence(file_path):
    midi_stream = converter.parse(file_path)
    notes_to_parse = midi_stream.flat.notesAndRests
    
    note_sequence = []
    for element in notes_to_parse:
        if isinstance(element, chord.Chord):
            note_sequence.append('.'.join(str(n.pitch.midi) for n in element.notes))
        elif isinstance(element, note.Note):
            note_sequence.append(str(element.pitch.midi))
        elif isinstance(element, note.Rest):
            note_sequence.append(str(element.name))
    
    return note_sequence
```

**Key Characteristics:**
- **Representation**: MIDI numbers for notes, dot-separated for chords, named rests
- **Tokenization**: Keras Tokenizer with character-level processing
- **Augmentation**: Pitch transposition (±1, ±2 semitones) for composers with <20 files
- **Sequence Length**: 95th percentile-based MAX_LEN determination

#### 3.2.2 Model Architecture
```python
model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=100, input_shape=(MAX_LEN,)),
    Conv1D(filters=128, kernel_size=5, activation='relu'),
    GlobalMaxPooling1D(),
    Dropout(0.5),
    RepeatVector(1),
    LSTM(128, return_sequences=False),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])
```

**Architecture Specifications:**
- **Embedding Dimension**: 100
- **CNN Filters**: 128 (kernel size: 5)
- **Pooling**: GlobalMaxPooling1D
- **LSTM Units**: 128
- **Regularization**: Dropout (0.5)
- **Training**: 50 epochs, batch size 64

### 3.3 Sarang's Alternate Model (MidiTok Approach)

#### 3.3.1 Data Preprocessing
```python
# MidiTok Configuration
config = TokenizerConfig(
    pitch_range=(21, 109), 
    beat_res={(0, 4): 8, (4, 12): 4},
    use_velocities=True,
    use_programs=True,
    use_time_signatures=True,
    use_tempos=True
)
tokenizer = REMI(tokenizer_config=config)
```

**Key Characteristics:**
- **Representation**: REMI (Revamped MIDI) tokenization
- **Features**: Pitch, velocity, program, time signatures, tempo
- **Tokenization**: Trained MidiTok vocabulary (10,000 tokens)
- **Augmentation**: Multi-modal (pitch shift, time stretch, velocity, dropout)

#### 3.3.2 Model Architecture
```python
model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=256, input_length=max_len),
    Conv1D(filters=128, kernel_size=5, activation='relu'),
    MaxPooling1D(pool_size=2),
    Conv1D(filters=64, kernel_size=3, activation='relu'),
    MaxPooling1D(pool_size=2),
    Bidirectional(LSTM(128, return_sequences=True)),
    Bidirectional(LSTM(64)),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(num_classes, activation='softmax')
])
```

**Architecture Specifications:**
- **Embedding Dimension**: 256
- **CNN Layers**: Dual-layer (128→64 filters)
- **LSTM**: Bidirectional, hierarchical (128→64 units)
- **Dense Layers**: Multi-layer (256→128 units)
- **Training**: 30 epochs, batch size 32

## 4. Results and Analysis

### 4.1 Performance Comparison

| Metric | Group 2 Model | Sarang's Alternate Model |
|--------|---------------|--------------------------|
| **Test Accuracy** | ~65% | ~66% |
| **Training Accuracy** | ~99% | ~99.9% |
| **Validation Accuracy** | ~25-34% | ~25-34% |
| **Overfitting Severity** | High | Very High |
| **Model Complexity** | Moderate | High |

### 4.2 Per-Composer Performance Analysis

#### 4.2.1 Group 2 Model Results
- **High Performance**: Bach (100%), Byrd (100%)
- **Moderate Performance**: Mozart (75%), Hummel/Handel/Mendelssohn (50%)
- **Low Performance**: Schumann (33%), Chopin (25%), Bartok (25%)

#### 4.2.2 Sarang's Alternate Model Results
- **High Performance**: Similar pattern with Bach and Byrd
- **Improved Recognition**: Enhanced feature extraction for some composers
- **Persistent Challenges**: Romantic period composers (Chopin, Schumann)

### 4.3 Overfitting Analysis

Both models exhibit severe overfitting characteristics:

**Evidence:**
- Training accuracy: >99%
- Validation accuracy: <35%
- Diverging loss curves after 20-30 epochs
- Poor generalization to test data

**Contributing Factors:**
1. **Limited Dataset Size**: Insufficient samples per composer
2. **High Model Complexity**: Deep architectures relative to data size
3. **Inadequate Regularization**: Current dropout rates insufficient
4. **Feature Redundancy**: Potential over-representation of certain patterns

## 5. Comparative Analysis

### 5.1 Model Design Comparison

| Aspect | Group 2 Model | Sarang's Alternate Model |
|--------|---------------|--------------------------|
| **Feature Extraction** | Basic symbolic representation | Advanced semantic tokenization |
| **Vocabulary Size** | Variable (text-based) | Fixed (10,000 tokens) |
| **Architecture Depth** | Moderate (5 layers) | Deep (8+ layers) |
| **Bidirectionality** | None | Bidirectional LSTM |
| **Augmentation Strategy** | Simple pitch transposition | Multi-modal augmentation |
| **Computational Complexity** | Lower | Higher |

### 5.2 Strengths and Weaknesses

#### 5.2.1 Group 2 Model
**Strengths:**
- Computational efficiency
- Interpretable feature representation
- Simpler architecture reduces overfitting risk
- Faster training and inference

**Weaknesses:**
- Limited musical feature capture
- Basic augmentation strategy
- Potential information loss in representation
- Less sophisticated temporal modeling

#### 5.2.2 Sarang's Alternate Model
**Strengths:**
- Rich musical feature representation
- Advanced tokenization preserves musical semantics
- Sophisticated augmentation techniques
- Bidirectional temporal modeling

**Weaknesses:**
- High computational requirements
- Increased overfitting susceptibility
- Complex architecture harder to interpret
- Longer training times

## 6. Potential Enhancements to Reduce Overfitting

### 6.1 Regularization Strategies

#### 6.1.1 Enhanced Dropout
```python
# Proposed improvement
model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=128),
    Dropout(0.2),  # Early dropout
    Conv1D(filters=64, kernel_size=5, activation='relu'),
    Dropout(0.3),
    MaxPooling1D(pool_size=2),
    LSTM(64, return_sequences=False),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dropout(0.4),
    Dense(num_classes, activation='softmax')
])
```

#### 6.1.2 Batch Normalization
```python
from tensorflow.keras.layers import BatchNormalization

# Add after each major layer
Conv1D(filters=64, kernel_size=5, activation='relu'),
BatchNormalization(),
```

#### 6.1.3 L1/L2 Regularization
```python
from tensorflow.keras.regularizers import l1_l2

Dense(64, activation='relu', 
      kernel_regularizer=l1_l2(l1=0.01, l2=0.01))
```

### 6.2 Data Augmentation Enhancements

#### 6.2.1 Advanced Augmentation Pipeline
```python
def enhanced_augmentation(midi_file, tokenizer):
    augmentations = []
    
    # Temporal augmentations
    augmentations.extend(time_stretch_augmentation(midi_file, [0.8, 0.9, 1.1, 1.2]))
    
    # Pitch augmentations with constraints
    augmentations.extend(pitch_shift_augmentation(midi_file, [-3, -2, -1, 1, 2, 3]))
    
    # Dynamic augmentations
    augmentations.extend(velocity_augmentation(midi_file, [-30, -15, 15, 30]))
    
    # Structural augmentations
    augmentations.extend(note_dropout_augmentation(tokens, [0.05, 0.1, 0.15, 0.2]))
    
    return augmentations
```

#### 6.2.2 Cross-Composer Style Transfer
```python
def style_transfer_augmentation(source_composer, target_style):
    # Apply style characteristics from one composer to another
    # Useful for increasing diversity while maintaining composer identity
    pass
```

### 6.3 Architecture Modifications

#### 6.3.1 Residual Connections
```python
def residual_block(x, filters):
    shortcut = x
    x = Conv1D(filters, 3, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv1D(filters, 3, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Add()([x, shortcut])
    return x
```

#### 6.3.2 Attention Mechanisms
```python
from tensorflow.keras.layers import MultiHeadAttention

# Add attention layer
attention_output = MultiHeadAttention(
    num_heads=8, 
    key_dim=64
)(lstm_output, lstm_output)
```

### 6.4 Training Strategies

#### 6.4.1 Early Stopping with Patience
```python
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=15,
    restore_best_weights=True,
    min_delta=0.001
)
```

#### 6.4.2 Learning Rate Scheduling
```python
from tensorflow.keras.callbacks import ReduceLROnPlateau

lr_scheduler = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=10,
    min_lr=1e-7
)
```

#### 6.4.3 Cross-Validation
```python
from sklearn.model_selection import StratifiedKFold

# Implement k-fold cross-validation
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
```

### 6.5 Ensemble Methods

#### 6.5.1 Model Averaging
```python
def ensemble_prediction(models, X_test):
    predictions = []
    for model in models:
        pred = model.predict(X_test)
        predictions.append(pred)
    
    # Average predictions
    ensemble_pred = np.mean(predictions, axis=0)
    return ensemble_pred
```

#### 6.5.2 Stacking
```python
# Train meta-learner on model outputs
meta_features = np.column_stack([
    model1.predict(X_val),
    model2.predict(X_val)
])
meta_model.fit(meta_features, y_val)
```

## 7. Discussion

### 7.1 Key Findings

1. **Feature Representation Impact**: MidiTok's advanced tokenization provides richer musical representation but doesn't necessarily translate to better generalization
2. **Architecture Complexity Trade-off**: More complex models achieve higher training accuracy but suffer from severe overfitting
3. **Composer-Specific Patterns**: Baroque composers (Bach, Byrd) are more easily distinguishable than Romantic period composers
4. **Augmentation Effectiveness**: Current augmentation strategies are insufficient to prevent overfitting

### 7.2 Implications for Music Information Retrieval

The results highlight fundamental challenges in music composer classification:
- **Data Scarcity**: Limited training samples per composer
- **Style Overlap**: Temporal proximity of composers leads to similar musical characteristics
- **Feature Engineering**: Need for more sophisticated feature extraction methods

### 7.3 Recommendations for Future Work

1. **Dataset Expansion**: Collect larger, more diverse datasets
2. **Transfer Learning**: Leverage pre-trained music models
3. **Multi-Modal Approaches**: Combine audio and symbolic representations
4. **Attention Mechanisms**: Implement attention to focus on discriminative patterns
5. **Regularization Research**: Investigate music-specific regularization techniques

## 8. Conclusion

This comparative analysis reveals that while both approaches achieve similar performance levels, they face common challenges related to overfitting and generalization. The MidiTok-based approach offers superior feature representation capabilities but requires more sophisticated regularization strategies. The traditional approach provides computational efficiency but may be limited by its simpler feature extraction methodology.

Future research should focus on developing music-specific regularization techniques, expanding datasets, and exploring ensemble methods to improve generalization capabilities. The integration of attention mechanisms and transfer learning approaches may provide pathways to more robust composer classification systems.

## References

1. Huang, C. Z. A., et al. (2018). Music transformer: Generating music with long-term structure. *arXiv preprint arXiv:1809.04281*.

2. Fradet, N., et al. (2023). MidiTok: A Python package for MIDI file tokenization. *Proceedings of the 24th International Society for Music Information Retrieval Conference*.

3. Cuthbert, M. S., & Ariza, C. (2010). music21: A toolkit for computer-aided musicology and symbolic music data. *Proceedings of the 11th International Society for Music Information Retrieval Conference*.

4. Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press.

5. Briot, J. P., Hadjeres, G., & Pachet, F. D. (2017). *Deep learning techniques for music generation*. Springer.
