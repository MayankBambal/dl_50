## **To Do**

**Level 1 (Product Manager Study):**
- **Video:**
- **Blog:**

**Level 2 (Junior Data Scientist):**
- **Code:**
  - [day25.ipynb](notebooks/day25.ipynb)
- **Books:**
- **Interview questions:**
  - [Easy Interview Questions](interview_questions/easy_questions.md)

**Level 3 (Data Scientist):**
- **Interview questions:**
  - [Medium Interview Questions](interview_questions/medium_questions.md)
- **Books:**

## **Plan**

**Objectives:**
- Understand LSTM architecture and how it solves vanishing gradients
- Learn about LSTM gates (forget, input, output)
- Implement LSTM in PyTorch
- Understand cell state vs. hidden state
- Learn when to use LSTMs

**Deep Learning Concept(s):**
- LSTM architecture: cell state and hidden state
- Forget gate: decides what to forget
- Input gate: decides what new information to store
- Output gate: decides what parts of cell state to output
- Cell state: long-term memory pathway
- How LSTMs solve vanishing gradient problem

**Tools Used:**
- PyTorch: `torch.nn.LSTM()`, `torch.nn.LSTMCell()`
- Functions: `nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)`

**Key Learnings:**
- Creating LSTM: `self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)`
- LSTM forward: `output, (hidden, cell) = lstm(input, (hidden, cell))`
- Understanding gates: forget gate, input gate, output gate, candidate values
- Cell state shape: `(num_layers, batch_size, hidden_size)`
- LSTM advantages: better at learning long-term dependencies
- When to use LSTM: sequence modeling, time series, NLP
- Understanding that LSTM has more parameters than RNN
- Bidirectional LSTMs: `bidirectional=True`

**References:**
- **Paper**: Hochreiter, S., & Schmidhuber, J. (1997). "Long Short-Term Memory"
- **Book**: "Deep Learning" by Ian Goodfellow - Chapter 10.10 (Long Short-Term Memory)
- **Tutorial**: [PyTorch: LSTM Tutorial](https://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html)
- **Visual Guide**: [Understanding LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)

---