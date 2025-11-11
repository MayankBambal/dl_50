## **To Do**

**Level 1 (Product Manager Study):**
- **Video:**
- **Blog:**

**Level 2 (Junior Data Scientist):**
- **Code:**
  - [day30.ipynb](notebooks/day30.ipynb)
- **Books:**
- **Interview questions:**
  - [Easy Interview Questions](interview_questions/easy_questions.md)

**Level 3 (Data Scientist):**
- **Interview questions:**
  - [Medium Interview Questions](interview_questions/medium_questions.md)
- **Books:**

## **Plan**

**Objectives:**
- Learn about GloVe embeddings
- Understand fastText for subword embeddings
- Compare different embedding methods
- Use pre-trained GloVe embeddings
- Understand when to use different embeddings

**Deep Learning Concept(s):**
- GloVe: Global Vectors for Word Representation
- fastText: subword-level embeddings (handles OOV words)
- Embedding comparison: Word2Vec vs. GloVe vs. fastText
- Global vs. local context
- Handling out-of-vocabulary words

**Tools Used:**
- fasttext: `fasttext` library
- Loading GloVe embeddings
- Functions: Using different embedding types in models

**Key Learnings:**
- GloVe: based on global co-occurrence statistics
- fastText: handles OOV words by using subword information
- Loading GloVe: converting GloVe format to PyTorch embeddings
- fastText advantages: works with misspellings, rare words
- When to use GloVe: when you want pre-trained, high-quality embeddings
- When to use fastText: when you have OOV problems, multiple languages
- Understanding that modern models use learned embeddings (contextual)
- Transition: from static embeddings to contextual embeddings (BERT/GPT)

**References:**
- **Paper**: Pennington, J., et al. (2014). "GloVe: Global Vectors for Word Representation"
- **Paper**: Bojanowski, P., et al. (2017). "Enriching Word Vectors with Subword Information"
- **Tutorial**: [GloVe Embeddings](https://nlp.stanford.edu/projects/glove/)
- **Tutorial**: [fastText Tutorial](https://fasttext.cc/docs/en/python-module.html)
- **Online**: [Word Embeddings Comparison](https://towardsdatascience.com/word-embeddings-comparison-glove-vs-word2vec-vs-fasttext-48eac9a61b6e)

---