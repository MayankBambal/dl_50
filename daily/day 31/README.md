## **To Do**

**Level 1 (Product Manager Study):**
- **Video:**
- **Blog:**

**Level 2 (Junior Data Scientist):**
- **Code:**
  - [day31.ipynb](notebooks/day31.ipynb)
- **Books:**
- **Interview questions:**
  - [Easy Interview Questions](interview_questions/easy_questions.md)

**Level 3 (Data Scientist):**
- **Interview questions:**
  - [Medium Interview Questions](interview_questions/medium_questions.md)
- **Books:**

## **Plan**

**Objectives:**
- Understand encoder-decoder (Seq2Seq) architecture
- Learn how encoder and decoder work together
- Implement encoder-decoder from scratch
- Understand the bottleneck representation
- Apply to machine translation

**Deep Learning Concept(s):**
- Encoder: converts input sequence to fixed-size representation
- Decoder: generates output sequence from representation
- Bottleneck: fixed-size vector (context vector)
- Sequence-to-sequence mapping
- Teacher forcing during training

**Tools Used:**
- PyTorch: building encoder and decoder with RNNs/LSTMs
- Functions: `nn.LSTM()` for encoder and decoder, managing hidden states

**Key Learnings:**
- Encoder: processes input sequence → final hidden state
- Decoder: takes encoder hidden state → generates output sequence
- Context vector: encoder's final hidden state
- Architecture: Encoder RNN → Context Vector → Decoder RNN
- Teacher forcing: feeding true previous token during training
- Inference: using predicted token as next input (autoregressive)
- Understanding the bottleneck problem (motivates attention, next day)
- Handling variable-length sequences

**References:**
- **Paper**: Sutskever, I., et al. (2014). "Sequence to Sequence Learning with Neural Networks"
- **Book**: "Deep Learning" by Ian Goodfellow - Chapter 10.4 (Encoder-Decoder Sequence-to-Sequence Architecture)
- **Tutorial**: [PyTorch: Seq2Seq Translation Tutorial](https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html)
- **Online**: [Understanding Encoder-Decoder Architecture](https://towardsdatascience.com/understanding-encoder-decoder-sequence-to-sequence-model-679e04af4346)

---