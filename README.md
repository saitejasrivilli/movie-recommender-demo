# Deep Learning Ad Recommender with Two-Stage Retrieval

A production-ready deep learning system for ad recommendation using two-stage retrieval: candidate generation with Two-Tower Neural Networks and ranking with Transformers.
#Demo: https://saitejasrivilli-movie-recommender-demo-app-phn9br.streamlit.app/
## 🎯 Overview

This project implements a state-of-the-art ad recommendation system that can:
- **Retrieve** from 1M+ ads in <50ms using FAISS
- **Rank** candidates using multi-head attention transformers
- **Optimize** for multiple objectives (CTR, engagement, revenue)
- **Scale** to production workloads with efficient architecture

### Architecture

```
┌─────────────┐
│   User      │
│  Features   │
└──────┬──────┘
       │
       ▼
┌─────────────────────────────────────┐
│  STAGE 1: Candidate Generation      │
│  ─────────────────────────────      │
│  Two-Tower Neural Network           │
│  • User Tower: Encode user features │
│  • Ad Tower: Encode ad features     │
│  • FAISS: Fast similarity search    │
│  1M ads → 500 candidates (<50ms)    │
└──────────┬──────────────────────────┘
           │
           ▼
┌─────────────────────────────────────┐
│  STAGE 2: Ranking                   │
│  ─────────────────────────           │
│  Transformer-based Ranker           │
│  • Multi-head attention             │
│  • Feature interactions             │
│  • Multi-task learning              │
│  500 candidates → 10 ads (50ms)     │
└──────────┬──────────────────────────┘
           │
           ▼
┌─────────────┐
│  Top 10     │
│  Ads        │
└─────────────┘
```

## 🚀 Quick Start

### Installation

```bash
# Clone and setup
cd /home/claude/ad_recommender

# Install dependencies
pip install torch torchvision --break-system-packages
pip install numpy pandas scikit-learn --break-system-packages
pip install faiss-cpu matplotlib tqdm --break-system-packages

# For GPU support (recommended)
pip install faiss-gpu --break-system-packages
```

### Train the Model

```bash
# Full training with synthetic data (quick demo)
python train.py \
    --use_synthetic \
    --n_samples 100000 \
    --stage1_epochs 5 \
    --stage2_epochs 5 \
    --batch_size 512

# Training with real Criteo data
python train.py \
    --data_path /path/to/criteo/train.txt \
    --n_samples 10000000 \
    --stage1_epochs 10 \
    --stage2_epochs 8 \
    --batch_size 2048 \
    --device cuda
```

### Run Inference

```bash
# Demo inference
python inference.py --demo

# Use in your application
from inference import AdRecommenderInference

recommender = AdRecommenderInference()
recommendations = recommender.recommend_ads(user_data, top_k=10)
```

## 📊 Features

### Stage 1: Two-Tower Neural Network
- **Separate Encoders**: Independent user and ad towers
- **Efficient Retrieval**: FAISS index for sub-50ms search
- **Contrastive Learning**: In-batch negative sampling
- **Scalability**: Can index millions of ads

### Stage 2: Transformer Ranker
- **Attention Mechanism**: Multi-head self-attention
- **Feature Interactions**: Cross-feature learning
- **Multi-Task Learning**: Optimize CTR, engagement, revenue simultaneously
- **Rich Context**: Incorporates user history and context

### FAISS Integration
- **Multiple Index Types**: Flat, IVF, IVFPQ, HNSW
- **GPU Support**: Accelerated search on GPU
- **Benchmark Tools**: Compare different index configurations
- **Production Ready**: Handles millions of vectors

## 📁 Project Structure

```
ad_recommender/
├── data/
│   └── synthetic_criteo.txt       # Synthetic training data
├── models/
│   ├── preprocessor.pkl           # Data preprocessor
│   ├── two_tower_best.pt          # Stage 1 model
│   ├── transformer_ranker_best.pt # Stage 2 model
│   ├── faiss_index.bin            # FAISS index
│   ├── two_tower_training.png     # Training curves
│   └── transformer_training.png   # Training curves
├── data_preprocessing.py          # Data preprocessing
├── two_tower_model.py             # Two-Tower architecture
├── transformer_ranker.py          # Transformer architecture
├── faiss_retrieval.py             # FAISS integration
├── training_pipeline.py           # Training utilities
├── train.py                       # Main training script
├── inference.py                   # Inference script
└── README.md                      # This file
```

## 🔧 Configuration

### Model Architecture

```python
# Two-Tower Model
user_tower = UserTower(
    user_feature_dims={...},      # User categorical features
    numerical_dim=13,              # Numerical features
    embedding_dim=16,              # Embedding size
    hidden_dims=[512, 256],        # Hidden layers
    output_dim=256,                # Embedding dimension
    dropout=0.3
)

# Transformer Ranker
ranker = TransformerRanker(
    d_model=256,                   # Model dimension
    num_heads=8,                   # Attention heads
    num_layers=3,                  # Transformer layers
    d_ff=1024,                     # Feed-forward dimension
    dropout=0.1
)
```

### Training Parameters

```python
# Stage 1 (Two-Tower)
stage1_epochs = 10
batch_size = 512
learning_rate = 0.001
loss = 0.5 * pointwise + 0.5 * contrastive

# Stage 2 (Transformer)
stage2_epochs = 8
batch_size = 512
learning_rate = 0.0001
loss = 1.0 * CTR + 0.5 * engagement + 0.3 * revenue
```

## 📈 Performance

### Speed Benchmarks

| Stage | Operation | Time | Throughput |
|-------|-----------|------|------------|
| 1 | Retrieve 500 from 1M ads | <50ms | 20 QPS |
| 2 | Rank 500 candidates | ~50ms | 20 QPS |
| **Total** | **End-to-end** | **<100ms** | **10+ QPS** |

### Model Quality — Measured (MovieLens 100K, NVIDIA A30)

Actual benchmark comparing SASRec (sequential attention) vs SVD (matrix factorization baseline)
on MovieLens 100K, leave-one-out protocol, 100-candidate pool.
Full results: [`benchmark_results.json`](benchmark_results.json)

| Model | HR@10 | NDCG@10 | Training Time |
|-------|-------|---------|---------------|
| SVD (baseline) | 0.3786 | 0.2124 | 6s |
| **SASRec** | **0.6638** | **0.3680** | ~690s |
| Gain | **+75.3%** | **+73.3%** | — |

SASRec best epoch: 56/60 (patience=8 early stopping). Dataset: 943 users, 1518 items after remapping.

## 🎓 Datasets

### Supported Datasets

1. **Criteo Display Advertising**
   - 45M+ click records
   - 13 numerical features
   - 26 categorical features
   - Download: [Kaggle](https://www.kaggle.com/c/criteo-display-ad-challenge)

2. **Outbrain Click Prediction**
   - 2B+ page views
   - Rich contextual features
   - Download: [Kaggle](https://www.kaggle.com/c/outbrain-click-prediction)

3. **Synthetic Data** (for testing)
   - Generated on-the-fly
   - Realistic feature distributions
   - Configurable size

### Data Format

```python
# Required format
features = {
    'label': 0/1,                    # Click label
    'I1-I13': numerical values,      # Numerical features
    'C1-C26': categorical values     # Categorical features
}
```

## 🔬 Advanced Usage

### Custom Feature Engineering

```python
from data_preprocessing import CriteoDataPreprocessor

preprocessor = CriteoDataPreprocessor(
    numerical_cols=['I1', 'I2', ...],
    categorical_cols=['C1', 'C2', ...]
)

# Add custom transformations
def custom_transform(df):
    # Your feature engineering
    return df

data = preprocessor.fit_transform(df)
```

### Hyperparameter Tuning

```python
# Grid search over key parameters
configs = [
    {'embedding_dim': 16, 'hidden_dims': [512, 256]},
    {'embedding_dim': 32, 'hidden_dims': [1024, 512, 256]},
    {'embedding_dim': 64, 'hidden_dims': [2048, 1024, 512]}
]

for config in configs:
    model = TwoTowerModel(**config)
    # Train and evaluate
```

### Production Deployment

```python
# Load models
recommender = AdRecommenderInference(
    model_dir='/path/to/models',
    device='cuda'  # Use GPU in production
)

# Serve recommendations
@app.route('/recommend')
def recommend():
    user_data = get_user_features(request)
    recs = recommender.recommend_ads(
        user_data,
        top_k=10,
        stage1_k=500
    )
    return jsonify(recs)
```

## 📊 Evaluation Metrics

### Retrieval Metrics (Stage 1)
- **Recall@k**: How many relevant ads in top-k
- **MRR**: Mean reciprocal rank
- **Hit Rate**: At least 1 relevant ad in top-k

### Ranking Metrics (Stage 2)
- **AUC**: Area under ROC curve
- **NDCG@k**: Normalized discounted cumulative gain
- **MAP@k**: Mean average precision

### System Metrics
- **Latency**: P50, P95, P99 response times
- **Throughput**: Queries per second
- **Index Size**: Memory footprint

## 🐛 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   ```python
   # Reduce batch size
   --batch_size 256
   
   # Use gradient accumulation
   accumulation_steps = 4
   ```

2. **FAISS Index Too Large**
   ```python
   # Use product quantization
   index_type = 'IVFPQ'
   
   # Reduce embedding dimension
   output_dim = 128
   ```

3. **Slow Training**
   ```python
   # Use mixed precision
   from torch.cuda.amp import autocast, GradScaler
   
   # Increase num_workers
   num_workers = 8
   ```

## SASRec: Sequential Recommendation

### Why Sequential Models Outperform Collaborative Filtering

SVD treats all of a user's interactions equally. SASRec weights recent interactions more heavily via attention, capturing the drift in user preferences over time. A user who watched action movies in 2020 and switched to documentaries in 2024 will get documentary recommendations from SASRec — SVD would still split the difference.

Concretely: SVD decomposes the user-item co-occurrence matrix into latent factors that capture *which* items a user likes, ignoring *when*. SASRec models the interaction sequence with causal self-attention, so recent items contribute more to the next-item prediction than older ones.

### Architecture (`models/sasrec.py`)

```
User history: [movie_1, movie_2, ..., movie_L]
                        │
              Item + Positional Embeddings
                        │
           ┌────────────────────────────┐
           │  SASRec Block × 2          │
           │  ┌─────────────────────┐   │
           │  │ Causal Self-Attention│   │  ← position i attends only to j ≤ i
           │  │ (pre-LayerNorm)      │   │
           │  └─────────────────────┘   │
           │  ┌─────────────────────┐   │
           │  │  FFN (4× hidden)    │   │
           │  │  (pre-LayerNorm)    │   │
           │  └─────────────────────┘   │
           └────────────────────────────┘
                        │
               Last position output
                        │
             Dot-product with item embeddings
                        │
                Next-item scores
```

- **`n_items`**: vocabulary size (number of movies)
- **`hidden_dim=64`**: embedding dimension
- **`n_heads=2`**: multi-head attention heads
- **`n_layers=2`**: transformer blocks
- **`max_seq_len=50`**: maximum history length, left-padded with zeros

### Benchmark vs SVD on MovieLens 100K

Evaluation: leave-one-out protocol — last item per user = test, 99 random negatives, rank among 100.

| Model  | HR@10  | NDCG@10 | Training Time | Notes                                                                 |
|--------|--------|---------|---------------|-----------------------------------------------------------------------|
| SVD    | 0.3786 | 0.2124  | 6s            | Matrix factorization baseline                                         |
| SASRec | 0.6638 | 0.3680  | ~690s         | Sequential model, best epoch 56/60 (patience=8), +75% HR@10 over SVD |

*Measured on NVIDIA A30. Full results in `benchmark_results.json`.*

Run the benchmark yourself:

```bash
python benchmark_sasrec_vs_svd.py
```

Train SASRec standalone (60 epochs with early stopping, reports HR@10 and NDCG@10 per epoch):

```bash
python train_sasrec.py
```

### Training Details (`train_sasrec.py`)

- **Dataset**: MovieLens 100K (100K ratings, 943 users, 1682 movies)
- **Feedback**: implicit — any rating = interaction
- **Sequences**: sorted by timestamp per user, max 50 items
- **Loss**: BCE with 1 positive + 99 random negatives per training step
- **Optimizer**: Adam, lr=1e-3, weight decay=1e-5
- **Epochs**: 60 (early stopping, patience=8)
- **Best epoch**: 56 (HR@10=0.6638, NDCG@10=0.3680)
- **Evaluation**: HR@10, NDCG@10 on held-out last item

### References

- Kang, W.-C., & McAuley, J. (2018). *Self-Attentive Sequential Recommendation*. ICDM 2018. [arXiv:1808.09781](https://arxiv.org/abs/1808.09781)

---

## 📚 References

### Papers
1. [Two Tower Models for Recommendations](https://research.google/pubs/pub47959/)
2. [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
3. [Deep Neural Networks for YouTube Recommendations](https://research.google/pubs/pub45530/)
4. [Wide & Deep Learning](https://arxiv.org/abs/1606.07792)

### Libraries
- [PyTorch](https://pytorch.org/)
- [FAISS](https://github.com/facebookresearch/faiss)
- [Scikit-learn](https://scikit-learn.org/)

## 🤝 Contributing

Contributions welcome! Areas for improvement:
- Additional dataset support
- More ranking architectures
- Distributed training
- Online learning
- A/B testing framework

## 📄 License

MIT License - feel free to use in your projects!

## 🙏 Acknowledgments

- Criteo for the public dataset
- Facebook AI for FAISS
- PyTorch team for the framework
- All contributors to open-source ML

---

**Built with ❤️ for the ML community**

For questions or issues, please open a GitHub issue or contact the maintainers.
