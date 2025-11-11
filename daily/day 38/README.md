## **To Do**

**Level 1 (Product Manager Study):**
- **Video:**
- **Blog:**

**Level 2 (Junior Data Scientist):**
- **Code:**
  - [day38.ipynb](notebooks/day38.ipynb)
- **Books:**
- **Interview questions:**
  - [Easy Interview Questions](interview_questions/easy_questions.md)

**Level 3 (Data Scientist):**
- **Interview questions:**
  - [Medium Interview Questions](interview_questions/medium_questions.md)
- **Books:**

## **Plan**

**Objectives:**
- Understand why multiple attention heads are used
- Learn how multi-head attention works
- Implement multi-head attention
- Understand head diversity
- Learn concatenation and projection

**Deep Learning Concept(s):**
- Multi-head attention: multiple attention mechanisms in parallel
- Different heads learn different relationships
- Head concatenation and linear projection
- Why multiple heads help
- Attention head specialization

**Tools Used:**
- PyTorch: `torch.nn.MultiheadAttention()`
- Manual implementation: splitting into heads, processing, concatenating

**Key Learnings:**
- Multi-head attention: `MultiHead(Q, K, V) = Concat(head_1, ..., head_h)W^O`
- Creating MultiheadAttention: `nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)`
- Splitting into heads: `d_model = num_heads * d_k`
- Each head: `(batch, num_heads, seq_len, d_k)` after splitting
- Concatenating heads: `(batch, seq_len, d_model)` then linear projection
- Understanding that different heads attend to different patterns
- Using PyTorch's built-in: `attn_output, attn_weights = multihead_attn(q, k, v)`
- Number of heads: typically 8, 16, or model_dim divisible

**References:**
- **Paper**: Vaswani, A., et al. (2017). "Attention is All You Need" - Section 3.2.2
- **Tutorial**: [PyTorch: MultiheadAttention](https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html)
- **Visual Guide**: [Multi-Head Attention](https://jalammar.github.io/illustrated-transformer/)
- **Online**: [Understanding Multi-Head Attention](https://towardsdatascience.com/illustrated-self-attention-2d627e33b20a)

---