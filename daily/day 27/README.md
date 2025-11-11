## **To Do**

**Level 1 (Product Manager Study):**
- **Video:**
- **Blog:**

**Level 2 (Junior Data Scientist):**
- **Code:**
  - [day27.ipynb](notebooks/day27.ipynb)
- **Books:**
- **Interview questions:**
  - [Easy Interview Questions](interview_questions/easy_questions.md)

**Level 3 (Data Scientist):**
- **Interview questions:**
  - [Medium Interview Questions](interview_questions/medium_questions.md)
- **Books:**

## **Plan**

**Objectives:**
- Understand bidirectional RNNs and their benefits
- Learn how bidirectional LSTMs process sequences
- Implement bidirectional LSTM in PyTorch
- Understand forward and backward passes
- Learn when bidirectional models are useful

**Deep Learning Concept(s):**
- Bidirectional processing: forward and backward passes
- Concatenating forward and backward hidden states
- Context from both past and future
- When bidirectional is beneficial
- Architecture differences

**Tools Used:**
- PyTorch: `torch.nn.LSTM(bidirectional=True)`
- Functions: `nn.LSTM(..., bidirectional=True)`

**Key Learnings:**
- Bidirectional LSTM: `nn.LSTM(..., bidirectional=True)`
- Output size: `hidden_size * 2` (forward + backward concatenated)
- Understanding forward and backward hidden states
- When to use bidirectional: when you have access to full sequence (classification, not generation)
- Not for causal tasks: can't use future info in real-time prediction
- Combining forward and backward: concatenation or other methods
- Use cases: sentiment analysis, named entity recognition, sequence classification
- Understanding that bidirectional doubles parameters

**References:**
- **Paper**: Schuster, M., & Paliwal, K. K. (1997). "Bidirectional recurrent neural networks"
- **Book**: "Deep Learning" by Ian Goodfellow - Chapter 10.3 (Bidirectional RNNs)
- **Tutorial**: [Bidirectional LSTM Tutorial](https://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html)
- **Online**: [Understanding Bidirectional RNNs](https://towardsdatascience.com/understanding-bidirectional-rnn-in-pytorch-5bd25a5dd66)

---