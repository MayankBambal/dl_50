## **To Do**

**Level 1 (Product Manager Study):**
- **Video:**
- **Blog:**

**Level 2 (Junior Data Scientist):**
- **Code:**
  - [day33.ipynb](notebooks/day33.ipynb)
- **Books:**
- **Interview questions:**
  - [Easy Interview Questions](interview_questions/easy_questions.md)

**Level 3 (Data Scientist):**
- **Interview questions:**
  - [Medium Interview Questions](interview_questions/medium_questions.md)
- **Books:**

## **Plan**

**Objectives:**
- Understand the attention mechanism
- Learn how attention weights work
- Implement attention from scratch
- Understand query, key, and value
- Apply attention to Seq2Seq models

**Deep Learning Concept(s):**
- Attention mechanism: dynamically focusing on relevant parts
- Attention weights: learned importance scores
- Query, Key, Value (Q, K, V) framework
- Attention scores: dot product or additive
- Weighted context vector (not fixed)
- Alignment between input and output

**Tools Used:**
- PyTorch: implementing attention mechanism manually
- Functions: computing attention scores, applying weights, creating context vectors

**Key Learnings:**
- Attention computation: `attention_scores = Q @ K.T / sqrt(d_k)`
- Softmax over attention scores: `attention_weights = softmax(attention_scores)`
- Weighted context: `context = attention_weights @ V`
- Attention in Seq2Seq: decoder attends to all encoder states
- Each decoder step gets different context vector (not fixed!)
- Understanding query (decoder), key (encoder), value (encoder)
- Masked attention: preventing looking at future tokens
- Scaled dot-product attention formula

**References:**
- **Paper**: Bahdanau, D., et al. (2014). "Neural Machine Translation by Jointly Learning to Align and Translate"
- **Paper**: Luong, M. T., et al. (2015). "Effective Approaches to Attention-based Neural Machine Translation"
- **Book**: "Deep Learning" by Ian Goodfellow - Chapter 12.4.5 (Attention Mechanisms)
- **Tutorial**: [Implementing Attention](https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html)
- **Visual Guide**: [Attention Mechanism Explained](https://lilianweng.github.io/posts/2018-06-24-attention/)

---