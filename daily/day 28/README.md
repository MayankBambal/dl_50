## **To Do**

**Level 1 (Product Manager Study):**
- **Video:**
- **Blog:**

**Level 2 (Junior Data Scientist):**
- **Code:**
  - [day28.ipynb](notebooks/day28.ipynb)
- **Books:**
- **Interview questions:**
  - [Easy Interview Questions](interview_questions/easy_questions.md)

**Level 3 (Data Scientist):**
- **Interview questions:**
  - [Medium Interview Questions](interview_questions/medium_questions.md)
- **Books:**

## **Plan**

**Objectives:**
- Apply LSTMs to a real NLP task (sentiment analysis)
- Preprocess text data (tokenization, vocabulary)
- Build a complete sentiment analysis model
- Train and evaluate the model
- Understand embedding layers (introduction)

**Deep Learning Concept(s):**
- Sentiment analysis: binary or multi-class text classification
- Text preprocessing pipeline
- Embedding layers: converting words to vectors
- Sequence classification with RNNs
- Using last hidden state for classification

**Tools Used:**
- PyTorch: `torch.nn.Embedding()`, `torch.nn.LSTM()`, `torch.nn.Linear()`
- Text preprocessing: tokenization libraries (basic)
- Functions: `nn.Embedding(vocab_size, embedding_dim)`, building complete NLP model

**Key Learnings:**
- Embedding layer: `self.embedding = nn.Embedding(vocab_size, embedding_dim)`
- Text preprocessing: tokenization, building vocabulary, converting to indices
- Model architecture: Embedding → LSTM → Linear → Output
- Using last hidden state: `output, (hidden, cell) = lstm(...)` then `hidden[-1]` for classification
- Padding sequences for batching
- Building vocabulary from training data
- Handling unknown words (UNK token)
- Understanding that embeddings will be covered in detail next week

**References:**
- **Book**: "Hands-On Machine Learning" by Aurélien Géron - Chapter 15 (Processing Sequences)
- **Tutorial**: [PyTorch: Sentiment Analysis Tutorial](https://pytorch.org/tutorials/beginner/text_sentiment_ngrams_tutorial.html)
- **Tutorial**: [Text Classification with PyTorch](https://pytorch.org/tutorials/beginner/nlp/deep_learning_tutorial.html)
- **Dataset**: [IMDB Movie Reviews](https://ai.stanford.edu/~amaas/data/sentiment/)

---