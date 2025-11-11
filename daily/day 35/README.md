## **To Do**

**Level 1 (Product Manager Study):**
- **Video:**
- **Blog:**

**Level 2 (Junior Data Scientist):**
- **Code:**
  - [day35.ipynb](notebooks/day35.ipynb)
- **Books:**
- **Interview questions:**
  - [Easy Interview Questions](interview_questions/easy_questions.md)

**Level 3 (Data Scientist):**
- **Interview questions:**
  - [Medium Interview Questions](interview_questions/medium_questions.md)
- **Books:**

## **Plan**

**Objectives:**
- Build a complete Seq2Seq model with attention
- Apply to machine translation or similar task
- Train and evaluate the model
- Compare with and without attention
- Understand practical considerations

**Deep Learning Concept(s):**
- Complete Seq2Seq pipeline with attention
- Machine translation workflow
- Evaluation metrics: BLEU score (introduction)
- Handling variable-length sequences
- Practical training considerations

**Tools Used:**
- PyTorch: complete Seq2Seq implementation
- Data: translation datasets
- Functions: Building end-to-end model with attention

**Key Learnings:**
- Complete architecture: Encoder (LSTM) → Attention → Decoder (LSTM) → Output
- Comparing models: with vs. without attention
- Understanding improvement from attention
- Handling special tokens: SOS, EOS, PAD
- Training considerations: teacher forcing, scheduled sampling
- Evaluation: BLEU score basics
- Preparing for transformers (which use self-attention, not just encoder-decoder attention)

**References:**
- **Paper**: Bahdanau, D., et al. (2014). "Neural Machine Translation by Jointly Learning to Align and Translate"
- **Tutorial**: [PyTorch: Complete Seq2Seq Tutorial](https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html)
- **Dataset**: [WMT Translation Dataset](http://www.statmt.org/wmt/)

---