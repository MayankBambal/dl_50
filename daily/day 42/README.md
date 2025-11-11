## **To Do**

**Level 1 (Product Manager Study):**
- **Video:**
- **Blog:**

**Level 2 (Junior Data Scientist):**
- **Code:**
  - [day42.ipynb](notebooks/day42.ipynb)
- **Books:**
- **Interview questions:**
  - [Easy Interview Questions](interview_questions/easy_questions.md)

**Level 3 (Data Scientist):**
- **Interview questions:**
  - [Medium Interview Questions](interview_questions/medium_questions.md)
- **Books:**

## **Plan**

**Objectives:**
- Review complete Transformer architecture
- Build full Transformer from scratch
- Understand all components working together
- Apply to a simple task
- Compare Transformer vs. previous architectures

**Deep Learning Concept(s):**
- Complete Transformer: Encoder + Decoder + Attention mechanisms
- End-to-end architecture
- Input/output processing
- Putting all pieces together

**Tools Used:**
- PyTorch: `torch.nn.Transformer()`
- Building complete model from scratch
- Functions: integrating all components

**Key Learnings:**
- Complete Transformer: `nn.Transformer(d_model, nhead, num_encoder_layers, num_decoder_layers, ...)`
- Input processing: embeddings + positional encoding
- Encoder: processes source sequence
- Decoder: generates target sequence with encoder attention
- Output: linear layer to vocabulary size
- Understanding full architecture end-to-end
- Comparison: Transformer vs. RNN/Seq2Seq (parallelization, performance)
- Preparing for BERT (encoder-only) and GPT (decoder-only)

**References:**
- **Paper**: Vaswani, A., et al. (2017). "Attention is All You Need"
- **Tutorial**: [PyTorch: Complete Transformer Tutorial](https://pytorch.org/tutorials/beginner/transformer_tutorial.html)
- **Visual Guide**: [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
- **Code**: [Transformer Implementation](https://github.com/pytorch/pytorch/blob/main/torch/nn/modules/transformer.py)

---