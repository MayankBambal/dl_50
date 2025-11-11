## **To Do**

**Level 1 (Product Manager Study):**
- **Video:**
- **Blog:**

**Level 2 (Junior Data Scientist):**
- **Code:**
  - [day41.ipynb](notebooks/day41.ipynb)
- **Books:**
- **Interview questions:**
  - [Easy Interview Questions](interview_questions/easy_questions.md)

**Level 3 (Data Scientist):**
- **Interview questions:**
  - [Medium Interview Questions](interview_questions/medium_questions.md)
- **Books:**

## **Plan**

**Objectives:**
- Understand why positional encodings are needed
- Learn sinusoidal positional encodings
- Implement positional encoding
- Understand learned vs. fixed encodings
- Learn relative positional encodings (introduction)

**Deep Learning Concept(s):**
- Why positional encodings: attention is permutation-invariant
- Sinusoidal encodings: fixed, mathematical patterns
- Learned positional embeddings
- Positional encoding addition to input embeddings
- Relative vs. absolute positions

**Tools Used:**
- PyTorch: implementing positional encodings
- Functions: creating sinusoidal patterns, learned embeddings

**Key Learnings:**
- Sinusoidal encoding: `PE(pos, 2i) = sin(pos / 10000^(2i/d_model))`, `PE(pos, 2i+1) = cos(...)`
- Adding to embeddings: `x = embedding + positional_encoding`
- Learned embeddings: `nn.Embedding(max_seq_len, d_model)` (learned)
- Fixed vs. learned: sinusoidal is fixed, embeddings are learned parameters
- Understanding that position info enables understanding of order
- Implementing: creating positional encoding matrix
- Positional encoding shape: `(max_seq_len, d_model)`
- Understanding that modern models often use learned positional embeddings

**References:**
- **Paper**: Vaswani, A., et al. (2017). "Attention is All You Need" - Section 3.5
- **Tutorial**: [Positional Encoding Implementation](https://pytorch.org/tutorials/beginner/transformer_tutorial.html)
- **Online**: [Understanding Positional Encodings](https://kazemnejad.com/blog/transformer_architecture_positional_encoding/)
- **Paper**: Shaw, P., et al. (2018). "Self-Attention with Relative Position Representations" (relative positions)

---