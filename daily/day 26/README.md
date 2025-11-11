## **To Do**

**Level 1 (Product Manager Study):**
- **Video:**
- **Blog:**

**Level 2 (Junior Data Scientist):**
- **Code:**
  - [day26.ipynb](notebooks/day26.ipynb)
- **Books:**
- **Interview questions:**
  - [Easy Interview Questions](interview_questions/easy_questions.md)

**Level 3 (Data Scientist):**
- **Interview questions:**
  - [Medium Interview Questions](interview_questions/medium_questions.md)
- **Books:**

## **Plan**

**Objectives:**
- Understand GRU architecture as a simpler alternative to LSTM
- Compare GRU vs. LSTM
- Implement GRU in PyTorch
- Understand reset gate and update gate
- Learn when to prefer GRU over LSTM

**Deep Learning Concept(s):**
- GRU architecture: simpler than LSTM
- Update gate: decides how much past information to keep
- Reset gate: decides how much past information to forget
- Single hidden state (no separate cell state)
- GRU vs. LSTM comparison
- Computational efficiency

**Tools Used:**
- PyTorch: `torch.nn.GRU()`, `torch.nn.GRUCell()`
- Functions: `nn.GRU(input_size, hidden_size, num_layers, batch_first=True)`

**Key Learnings:**
- Creating GRU: `self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)`
- GRU forward: `output, hidden = gru(input, hidden)`
- GRU has 2 gates vs. LSTM's 3 gates (simpler)
- GRU advantages: fewer parameters, faster training, often similar performance
- When to use GRU: when you want simpler model, less data
- When to use LSTM: when you need maximum capacity, very long sequences
- Understanding that GRU is a good default choice
- Bidirectional GRUs: `bidirectional=True`

**References:**
- **Paper**: Cho, K., et al. (2014). "Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation"
- **Book**: "Deep Learning" by Ian Goodfellow - Chapter 10.10 (Gated RNNs)
- **Tutorial**: [PyTorch: GRU Documentation](https://pytorch.org/docs/stable/generated/torch.nn.GRU.html)
- **Online**: [GRU vs. LSTM](https://towardsdatascience.com/understanding-gru-networks-2ef37df6c9be)

---