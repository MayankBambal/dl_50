## **To Do**

**Level 1 (Product Manager Study):**
- **Video:**
- **Blog:**

**Level 2 (Junior Data Scientist):**
- **Code:**
  - [day40.ipynb](notebooks/day40.ipynb)
- **Books:**
- **Interview questions:**
  - [Easy Interview Questions](interview_questions/easy_questions.md)

**Level 3 (Data Scientist):**
- **Interview questions:**
  - [Medium Interview Questions](interview_questions/medium_questions.md)
- **Books:**

## **Plan**

**Objectives:**
- Understand Transformer decoder architecture
- Learn masked self-attention
- Implement decoder layer
- Understand encoder-decoder attention
- Learn autoregressive generation

**Deep Learning Concept(s):**
- Decoder block: Masked Multi-Head Attention → Encoder-Decoder Attention → FFN
- Masked self-attention: preventing looking at future tokens
- Encoder-decoder attention: decoder queries attend to encoder outputs
- Autoregressive generation
- Causal masking

**Tools Used:**
- PyTorch: `torch.nn.TransformerDecoder()`, `torch.nn.TransformerDecoderLayer()`
- Functions: implementing causal masks

**Key Learnings:**
- Decoder layer: `nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout)`
- Masked self-attention: causal mask (lower triangular)
- Encoder-decoder attention: `decoder_output @ encoder_output.T`
- Building decoder: `decoder = nn.TransformerDecoder(decoder_layer, num_layers)`
- Causal mask: `torch.triu(torch.ones(seq_len, seq_len)) == 0`
- Understanding autoregressive: generating one token at a time
- Decoder uses both: masked self-attention (past tokens) + encoder attention (input)
- Inference: iterative generation with teacher forcing removed

**References:**
- **Paper**: Vaswani, A., et al. (2017). "Attention is All You Need" - Section 3.1
- **Tutorial**: [PyTorch: TransformerDecoder](https://pytorch.org/docs/stable/generated/torch.nn.TransformerDecoder.html)
- **Visual Guide**: [Transformer Decoder](https://jalammar.github.io/illustrated-transformer/)
- **Online**: [Understanding Transformer Decoder](https://towardsdatascience.com/how-does-the-transformer-decoder-work-6b682b1ef7a3)

---