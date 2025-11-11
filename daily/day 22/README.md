## **To Do**

**Level 1 (Product Manager Study):**
- **Video:**
- **Blog:**

**Level 2 (Junior Data Scientist):**
- **Code:**
  - [day22.ipynb](notebooks/day22.ipynb)
- **Books:**
- **Interview questions:**
  - [Easy Interview Questions](interview_questions/easy_questions.md)

**Level 3 (Data Scientist):**
- **Interview questions:**
  - [Medium Interview Questions](interview_questions/medium_questions.md)
- **Books:**

## **Plan**

**Objectives:**
- Understand what sequential data is and why it's different
- Learn about time series and text data
- Understand variable-length sequences
- Learn about padding and masking
- Prepare text data for neural networks

**Deep Learning Concept(s):**
- Sequential data characteristics
- Time dependencies
- Variable-length sequences
- Padding sequences to fixed length
- Masking padded tokens
- Tokenization basics

**Tools Used:**
- PyTorch: `torch.nn.utils.rnn.pad_sequence()`, `torch.nn.utils.rnn.pack_padded_sequence()`
- Functions: Manual padding, sequence length tracking

**Key Learnings:**
- Padding sequences: `pad_sequence(sequences, batch_first=True, padding_value=0)`
- Packing sequences: `pack_padded_sequence(padded, lengths, batch_first=True)`
- Understanding that sequences have temporal dependencies
- Text preprocessing: tokenization (introduction)
- Sequence length vs. batch dimension
- Masking: ignoring padding tokens in loss computation
- Understanding why we need special architectures for sequences

**References:**
- **Book**: "Deep Learning" by Ian Goodfellow - Chapter 10.1 (Recurrent Neural Networks - Introduction)
- **Book**: "Hands-On Machine Learning" by Aurélien Géron - Chapter 15 (Processing Sequences Using RNNs and CNNs)
- **Tutorial**: [PyTorch: Working with Variable Length Sequences](https://pytorch.org/tutorials/beginner/seq2seq_translation_tutorial.html)
- **Online**: [Understanding Sequential Data](https://towardsdatascience.com/understanding-rnn-and-lstm-f7cdf6dfc14e)

---