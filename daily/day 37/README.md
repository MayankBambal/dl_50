## **To Do**

**Level 1 (Product Manager Study):**
- **Video:**
- **Blog:**

**Level 2 (Junior Data Scientist):**
- **Code:**
  - [day37.ipynb](notebooks/day37.ipynb)
- **Books:**
- **Interview questions:**
  - [Easy Interview Questions](interview_questions/easy_questions.md)

**Level 3 (Data Scientist):**
- **Interview questions:**
  - [Medium Interview Questions](interview_questions/medium_questions.md)
- **Books:**

## **Plan**

**Objectives:**
- Understand self-attention mechanism in detail
- Learn Q, K, V computation
- Implement self-attention from scratch
- Understand scaled dot-product attention
- Learn attention masks

**Deep Learning Concept(s):**
- Self-attention: attending to all positions including itself
- Query, Key, Value computation
- Scaled dot-product attention formula
- Attention mask: masking future/padding tokens
- Attention weights and their interpretation

**Tools Used:**
- PyTorch: `torch.nn.functional.scaled_dot_product_attention()` (PyTorch 2.0+)
- Manual implementation: matrix multiplications for Q, K, V

**Key Learnings:**
- Self-attention: `Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V`
- Computing Q, K, V: `Q = XW_q`, `K = XW_k`, `V = XW_v` (from same input)
- Scaled dot-product: division by `sqrt(d_k)` prevents softmax saturation
- Attention mask: `masked_fill(mask == 0, float('-inf'))` before softmax
- Understanding attention matrix: `(seq_len, seq_len)` for each head
- Implementing from scratch vs. using PyTorch function
- Batch processing: handling `(batch, seq_len, d_model)` tensors

**References:**
- **Paper**: Vaswani, A., et al. (2017). "Attention is All You Need" - Section 3.2.1
- **Tutorial**: [PyTorch: Scaled Dot-Product Attention](https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)
- **Visual Guide**: [Self-Attention Explained](https://jalammar.github.io/illustrated-transformer/)
- **Video**: [Attention Mechanism in Detail](https://www.youtube.com/watch?v=rBCqOTEfxvg)

---