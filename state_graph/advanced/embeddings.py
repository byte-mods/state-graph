"""Embeddings — create, train, fine-tune, and serve custom embedding models.

Supports: text, image, audio, multimodal embeddings.
Train from scratch or fine-tune from pretrained (sentence-transformers, CLIP, etc.).
"""

from __future__ import annotations

import json
from typing import Any


EMBEDDING_MODELS = {
    "text": {
        "sentence-transformers/all-MiniLM-L6-v2": {"dim": 384, "description": "Fast, lightweight text embeddings"},
        "sentence-transformers/all-mpnet-base-v2": {"dim": 768, "description": "Best quality general-purpose"},
        "BAAI/bge-large-en-v1.5": {"dim": 1024, "description": "BGE large — MTEB top performer"},
        "BAAI/bge-m3": {"dim": 1024, "description": "Multilingual, multi-granularity"},
        "intfloat/e5-large-v2": {"dim": 1024, "description": "E5 embeddings — excellent retrieval"},
        "thenlper/gte-large": {"dim": 1024, "description": "General Text Embeddings"},
        "nomic-ai/nomic-embed-text-v1.5": {"dim": 768, "description": "Open-source with Matryoshka support"},
    },
    "image": {
        "openai/clip-vit-base-patch32": {"dim": 512, "description": "CLIP image embeddings"},
        "openai/clip-vit-large-patch14": {"dim": 768, "description": "CLIP large — better quality"},
        "google/siglip-base-patch16-224": {"dim": 768, "description": "SigLIP — improved CLIP"},
        "timm/vit_base_patch16_224": {"dim": 768, "description": "ViT feature extraction"},
    },
    "multimodal": {
        "openai/clip-vit-base-patch32": {"dim": 512, "description": "CLIP — joint image-text space"},
        "laion/CLIP-ViT-H-14-laion2B": {"dim": 1024, "description": "OpenCLIP huge — best quality"},
    },
}


def generate_embedding_training_script(
    base_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    task: str = "similarity",
    dataset: str = "",
    params: dict = None,
) -> str:
    """Generate script to train/fine-tune custom embeddings."""
    params = params or {}

    if task == "similarity":
        return f'''"""Fine-tune text embedding model for semantic similarity."""
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader

# Load base model
model = SentenceTransformer("{base_model}")

# Training data — pairs of (text_a, text_b, similarity_score)
# Score: 0.0 (unrelated) to 1.0 (identical meaning)
train_examples = [
    InputExample(texts=["cat", "kitten"], label=0.9),
    InputExample(texts=["cat", "car"], label=0.1),
    # Add your pairs here or load from file:
    # import json
    # for item in json.load(open("{dataset or 'data/pairs.json'}")):
    #     train_examples.append(InputExample(texts=[item["a"], item["b"]], label=item["score"]))
]

loader = DataLoader(train_examples, shuffle=True, batch_size={params.get('batch_size', 16)})
loss = losses.CosineSimilarityLoss(model)

model.fit(
    train_objectives=[(loader, loss)],
    epochs={params.get('epochs', 3)},
    warmup_steps={params.get('warmup', 100)},
    output_path="./custom_embeddings",
)

# Test
embeddings = model.encode(["Hello world", "Hi there"])
from sklearn.metrics.pairwise import cosine_similarity
print(f"Similarity: {{cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]:.4f}}")
'''

    elif task == "classification":
        return f'''"""Train embeddings for classification (SetFit — few-shot)."""
from setfit import SetFitModel, SetFitTrainer
from datasets import Dataset

model = SetFitModel.from_pretrained("{base_model}")

# Few-shot training data
train_data = Dataset.from_dict({{
    "text": ["Great product!", "Terrible service", "Love it", "Worst ever", "Amazing quality", "Broken on arrival"],
    "label": [1, 0, 1, 0, 1, 0],
}})

trainer = SetFitTrainer(
    model=model,
    train_dataset=train_data,
    num_iterations={params.get('iterations', 20)},
    batch_size={params.get('batch_size', 16)},
)

trainer.train()
model.save_pretrained("./custom_classifier_embeddings")

# Test
preds = model.predict(["This is wonderful", "This is awful"])
print(f"Predictions: {{preds}}")
'''

    elif task == "retrieval":
        return f'''"""Train embeddings for retrieval (query-document matching)."""
from sentence_transformers import SentenceTransformer, InputExample, losses
from sentence_transformers.evaluation import InformationRetrievalEvaluator
from torch.utils.data import DataLoader

model = SentenceTransformer("{base_model}")

# Training data — (query, positive_document) pairs
train_examples = [
    InputExample(texts=["What is Python?", "Python is a programming language..."]),
    InputExample(texts=["How to cook pasta?", "Boil water, add pasta, cook for 8 minutes..."]),
    # Load your data here
]

loader = DataLoader(train_examples, shuffle=True, batch_size={params.get('batch_size', 16)})
loss = losses.MultipleNegativesRankingLoss(model)  # Best for retrieval

model.fit(
    train_objectives=[(loader, loss)],
    epochs={params.get('epochs', 3)},
    output_path="./retrieval_embeddings",
)

# Build index for search
corpus = ["Document 1...", "Document 2...", "Document 3..."]
corpus_embeddings = model.encode(corpus)

query = "search query"
query_embedding = model.encode([query])

from sklearn.metrics.pairwise import cosine_similarity
scores = cosine_similarity(query_embedding, corpus_embeddings)[0]
top_idx = scores.argsort()[::-1][:5]
for i in top_idx:
    print(f"Score: {{scores[i]:.4f}} — {{corpus[i][:100]}}")
'''

    elif task == "from_scratch":
        return f'''"""Train embeddings from scratch with contrastive learning."""
import torch
import torch.nn as nn
import torch.nn.functional as F

class EmbeddingModel(nn.Module):
    def __init__(self, vocab_size={params.get('vocab_size', 30000)}, embed_dim={params.get('embed_dim', 256)}, hidden_dim={params.get('hidden_dim', 512)}):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=8, dim_feedforward=hidden_dim, batch_first=True),
            num_layers={params.get('num_layers', 4)},
        )
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, input_ids):
        x = self.embedding(input_ids)
        x = self.encoder(x)
        x = self.pool(x.transpose(1, 2)).squeeze(-1)
        return F.normalize(x, p=2, dim=-1)

# Contrastive loss (InfoNCE)
def contrastive_loss(anchor, positive, temperature=0.07):
    sim = torch.mm(anchor, positive.T) / temperature
    labels = torch.arange(len(anchor), device=anchor.device)
    return F.cross_entropy(sim, labels)

model = EmbeddingModel()
optimizer = torch.optim.AdamW(model.parameters(), lr={params.get('lr', 1e-4)})

# Training loop
for epoch in range({params.get('epochs', 10)}):
    # Replace with your paired data (anchor_ids, positive_ids)
    anchor_ids = torch.randint(0, {params.get('vocab_size', 30000)}, (32, 64))
    positive_ids = torch.randint(0, {params.get('vocab_size', 30000)}, (32, 64))

    anchor_emb = model(anchor_ids)
    positive_emb = model(positive_ids)
    loss = contrastive_loss(anchor_emb, positive_emb)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(f"Epoch {{epoch}}, Loss: {{loss.item():.4f}}")

torch.save(model.state_dict(), "./custom_embeddings.pt")
'''

    return "# Embedding training script\n"
