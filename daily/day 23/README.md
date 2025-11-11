## **To Do**

**Level 1 (Product Manager Study):**
- **Video:**
- **Blog:**

**Level 2 (Junior Data Scientist):**
- **Code:**
  - [day23.ipynb](notebooks/day23.ipynb)
- **Books:**
- **Interview questions:**
  - [Easy Interview Questions](interview_questions/easy_questions.md)

**Level 3 (Data Scientist):**
- **Interview questions:**
  - [Medium Interview Questions](interview_questions/medium_questions.md)
- **Books:**

## **Plan**

**Objectives:**
- Understand the architecture of Recurrent Neural Networks
- Learn how RNNs maintain hidden state
- Implement RNN from scratch
- Understand forward pass through time
- Learn to use PyTorch RNN modules

**Deep Learning Concept(s):**
- RNN architecture: hidden state propagation
- Recurrent connection: h_t = f(W_hh * h_{t-1} + W_xh * x_t + b)
- Unrolling RNN through time
- Sharing weights across time steps
- Hidden state as memory
- Backpropagation through time (BPTT)

**Tools Used:**
- PyTorch: `torch.nn.RNN()`, `torch.nn.RNNCell()`
- Functions: `nn.RNN(input_size, hidden_size, num_layers, batch_first=True)`

**Key Learnings:**
- Creating RNN: `self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)`
- RNN forward pass: `output, hidden = rnn(input, hidden)`
- Understanding hidden state shape: `(num_layers, batch_size, hidden_size)`
- Single vs. multi-layer RNNs
- Understanding that RNN processes sequences step by step
- Output shape: `(batch_size, seq_len, hidden_size)`
- Last hidden state vs. all outputs
- Bidirectional RNNs (introduction)

**References:**
- **Book**: "Deep Learning" by Ian Goodfellow - Chapter 10.2 (Recurrent Neural Networks)
- **Book**: "Neural Networks and Deep Learning" by Michael Nielsen - Chapter 5 (Deep Learning - Recurrent Neural Networks)
- **Tutorial**: [PyTorch: RNN Tutorial](https://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html)
- **Paper**: Elman, J. L. (1990). "Finding structure in time"

---