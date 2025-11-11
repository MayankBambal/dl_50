## **To Do**

**Level 1 (Product Manager Study):**
- **Video:**
- **Blog:**

**Level 2 (Junior Data Scientist):**
- **Code:**
  - [day49.ipynb](notebooks/day49.ipynb)
- **Books:**
- **Interview questions:**
  - [Easy Interview Questions](interview_questions/easy_questions.md)

**Level 3 (Data Scientist):**
- **Interview questions:**
  - [Medium Interview Questions](interview_questions/medium_questions.md)
- **Books:**

## **Plan**

**Objectives:**
- Compare encoder-only (BERT), decoder-only (GPT), and encoder-decoder (T5) architectures
- Understand when to use each architecture
- Learn about T5 (Text-to-Text Transfer Transformer)
- Understand task-specific model selection
- Learn about modern model families

**Deep Learning Concept(s):**
- Architecture comparison: encoder-only, decoder-only, encoder-decoder
- BERT: bidirectional, good for understanding tasks
- GPT: autoregressive, good for generation tasks
- T5: encoder-decoder, good for both understanding and generation
- Task-specific architectures
- Modern model families

**Tools Used:**
- Hugging Face: comparing different model architectures
- Understanding model selection

**Key Learnings:**
- BERT: best for classification, NER, QA (understanding tasks)
- GPT: best for text generation, completion (generation tasks)
- T5: unified architecture, all tasks as text-to-text (translation, summarization, classification as text)
- Understanding that modern models often use hybrid approaches
- Model families: BERT family (RoBERTa, ALBERT), GPT family (GPT-2, GPT-3, GPT-4), T5 family
- Task selection: choose architecture based on task type
- Understanding that some modern models blur these distinctions (decoder-only for everything)

**References:**
- **Paper**: Raffel, C., et al. (2019). "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer" (T5)
- **Tutorial**: [Hugging Face: Model Overview](https://huggingface.co/docs/transformers/model_summary)
- **Online**: [BERT vs. GPT vs. T5](https://towardsdatascience.com/bert-vs-gpt-vs-t5-a-comparison-of-the-three-major-models-12f6d7e3c3b5)
- **Online**: [Understanding Different Transformer Architectures](https://huggingface.co/docs/transformers/model_summary)

---