## **To Do**

**Level 1 (Product Manager Study):**
- **Video:**
- **Blog:**

**Level 2 (Junior Data Scientist):**
- **Code:**
  - [day39.ipynb](notebooks/day39.ipynb)
- **Books:**
- **Interview questions:**
  - [Easy Interview Questions](interview_questions/easy_questions.md)

**Level 3 (Data Scientist):**
- **Interview questions:**
  - [Medium Interview Questions](interview_questions/medium_questions.md)
- **Books:**

## **Plan**

**Objectives:**
- Understand Transformer encoder architecture
- Learn encoder block components
- Implement encoder layer
- Understand feed-forward networks
- Learn layer normalization and residual connections

**Deep Learning Concept(s):**
- Encoder block: Multi-Head Attention → Add & Norm → FFN → Add & Norm
- Feed-forward network: two linear layers with ReLU
- Residual connections (skip connections)
- Layer normalization
- Position-wise FFN
- Stacking encoder layers

**Tools Used:**
- PyTorch: `torch.nn.TransformerEncoder()`, `torch.nn.TransformerEncoderLayer()`
- Manual implementation: building encoder blocks

**Key Learnings:**
- Encoder layer: `nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)`
- Feed-forward: `FFN(x) = max(0, xW1 + b1)W2 + b2` (typically expands then contracts)
- Residual connection: `output = LayerNorm(x + Sublayer(x))`
- Layer normalization: `nn.LayerNorm(d_model)` (normalizes across features)
- Building encoder: `encoder = nn.TransformerEncoder(encoder_layer, num_layers)`
- Understanding that each encoder layer refines representations
- Dropout for regularization
- Understanding 6 encoder layers in original Transformer

**References:**
- **Paper**: Vaswani, A., et al. (2017). "Attention is All You Need" - Section 3.1
- **Tutorial**: [PyTorch: TransformerEncoder](https://pytorch.org/docs/stable/generated/torch.nn.TransformerEncoder.html)
- **Visual Guide**: [Transformer Encoder](https://jalammar.github.io/illustrated-transformer/)
- **Online**: [Transformer Architecture Deep Dive](https://towardsdatascience.com/transformer-architecture-explained-8bb59b4e16e8)

---