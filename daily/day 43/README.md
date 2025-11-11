## **To Do**

**Level 1 (Product Manager Study):**
- **Video:**
- **Blog:**

**Level 2 (Junior Data Scientist):**
- **Code:**
  - [day43.ipynb](notebooks/day43.ipynb)
- **Books:**
- **Interview questions:**
  - [Easy Interview Questions](interview_questions/easy_questions.md)

**Level 3 (Data Scientist):**
- **Interview questions:**
  - [Medium Interview Questions](interview_questions/medium_questions.md)
- **Books:**

## **Plan**

**Objectives:**
- Understand BERT as encoder-only Transformer
- Learn BERT architecture details
- Understand bidirectional context
- Learn about BERT variants (BERT-base, BERT-large)
- Compare BERT vs. original Transformer

**Deep Learning Concept(s):**
- BERT: Bidirectional Encoder Representations from Transformers
- Encoder-only architecture (no decoder)
- Bidirectional self-attention (can see full sequence)
- Layer stacking (12 or 24 layers)
- Pre-training vs. fine-tuning

**Tools Used:**
- Hugging Face Transformers: `transformers.BertModel`, `transformers.BertConfig`
- Functions: `BertModel.from_pretrained()`, understanding architecture

**Key Learnings:**
- Loading BERT: `from transformers import BertModel; model = BertModel.from_pretrained('bert-base-uncased')`
- BERT architecture: only Transformer encoder (no decoder)
- Bidirectional: unlike GPT, BERT sees full context both ways
- BERT variants: BERT-base (12 layers, 768 hidden), BERT-large (24 layers, 1024 hidden)
- Understanding that BERT is pre-trained, then fine-tuned
- BERT embeddings: contextual (different for same word in different contexts)
- Tokenizer: WordPiece tokenization
- Special tokens: [CLS], [SEP], [MASK], [PAD]

**References:**
- **Paper**: Devlin, J., et al. (2018). "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding"
- **Tutorial**: [Hugging Face: BERT Documentation](https://huggingface.co/docs/transformers/model_doc/bert)
- **Tutorial**: [Using BERT](https://huggingface.co/docs/transformers/training)
- **Visual Guide**: [BERT Explained](https://jalammar.github.io/illustrated-bert/)

---