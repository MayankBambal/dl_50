## **To Do**

**Level 1 (Product Manager Study):**
- **Video:**
- **Blog:**

**Level 2 (Junior Data Scientist):**
- **Code:**
  - [day44.ipynb](notebooks/day44.ipynb)
- **Books:**
- **Interview questions:**
  - [Easy Interview Questions](interview_questions/easy_questions.md)

**Level 3 (Data Scientist):**
- **Interview questions:**
  - [Medium Interview Questions](interview_questions/medium_questions.md)
- **Books:**

## **Plan**

**Objectives:**
- Understand how BERT is pre-trained
- Learn Masked Language Modeling (MLM) objective
- Implement MLM training
- Understand next sentence prediction (NSP)
- Learn pre-training vs. fine-tuning

**Deep Learning Concept(s):**
- Masked Language Modeling: predicting masked tokens
- Pre-training objective: learning language representations
- 15% masking strategy
- Next Sentence Prediction (NSP)
- Self-supervised learning

**Tools Used:**
- Hugging Face: `transformers.BertForMaskedLM`, `transformers.DataCollatorForLanguageModeling`
- Functions: masking tokens, training MLM model

**Key Learnings:**
- MLM: mask 15% of tokens, predict original tokens
- Masking strategy: 80% [MASK], 10% random token, 10% unchanged
- Loading MLM model: `BertForMaskedLM.from_pretrained('bert-base-uncased')`
- Data collator: `DataCollatorForLanguageModeling(tokenizer, mlm=True, mlm_probability=0.15)`
- Pre-training: learning on large unlabeled text (Wikipedia, BooksCorpus)
- Understanding self-supervised learning (no human labels needed)
- NSP: predicting if sentence B follows sentence A (binary classification)
- Fine-tuning: adapting pre-trained BERT to specific tasks

**References:**
- **Paper**: Devlin, J., et al. (2018). "BERT: Pre-training..." - Section 3.1, 3.3
- **Tutorial**: [Hugging Face: Masked Language Modeling](https://huggingface.co/docs/transformers/tasks/masked_language_modeling)
- **Tutorial**: [Pre-training BERT](https://huggingface.co/docs/transformers/training)
- **Online**: [Understanding BERT Pre-training](https://towardsdatascience.com/bert-explained-state-of-the-art-language-model-for-nlp-f8b21a9b6270)

---