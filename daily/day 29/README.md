## **To Do**

**Level 1 (Product Manager Study):**
- **Video:**
- **Blog:**

**Level 2 (Junior Data Scientist):**
- **Code:**
  - [day29.ipynb](notebooks/day29.ipynb)
- **Books:**
- **Interview questions:**
  - [Easy Interview Questions](interview_questions/easy_questions.md)

**Level 3 (Data Scientist):**
- **Interview questions:**
  - [Medium Interview Questions](interview_questions/medium_questions.md)
- **Books:**

## **Plan**

**Objectives:**
- Understand what word embeddings are
- Learn Word2Vec (Skip-gram and CBOW)
- Understand how embeddings capture semantic meaning
- Use pre-trained word embeddings
- Visualize word embeddings

**Deep Learning Concept(s):**
- Word embeddings: dense vector representations
- Word2Vec: predicting context (Skip-gram) or word from context (CBOW)
- Semantic relationships in embedding space
- Vector similarity: cosine similarity
- Pre-trained embeddings (GloVe, Word2Vec)

**Tools Used:**
- gensim: `gensim.models.Word2Vec`, loading pre-trained embeddings
- PyTorch: `torch.nn.Embedding()` with pre-trained weights
- Functions: `Word2Vec()`, loading embeddings, similarity calculations

**Key Learnings:**
- Using pre-trained embeddings: loading Word2Vec/GloVe vectors
- Loading into PyTorch: `nn.Embedding.from_pretrained(weights, freeze=True/False)`
- Understanding embedding dimensions: typically 100, 200, 300
- Semantic relationships: king - man + woman ≈ queen
- Finding similar words using cosine similarity
- Training your own embeddings (optional)
- Understanding that embeddings are learnable parameters
- Freezing vs. fine-tuning embeddings

**References:**
- **Paper**: Mikolov, T., et al. (2013). "Efficient Estimation of Word Representations in Vector Space"
- **Book**: "Deep Learning" by Ian Goodfellow - Chapter 12.4 (Word Embeddings)
- **Tutorial**: [Word2Vec Tutorial](https://radimrehurek.com/gensim/auto_examples/tutorials/run_word2vec.html)
- **Online**: [Understanding Word Embeddings](https://towardsdatascience.com/introduction-to-word-embedding-and-word2vec-652d0c2060fa)

---