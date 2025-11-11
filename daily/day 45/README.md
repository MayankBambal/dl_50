## **To Do**

**Level 1 (Product Manager Study):**
- **Video:**
- **Blog:**

**Level 2 (Junior Data Scientist):**
- **Code:**
  - [day45.ipynb](notebooks/day45.ipynb)
- **Books:**
- **Interview questions:**
  - [Easy Interview Questions](interview_questions/easy_questions.md)

**Level 3 (Data Scientist):**
- **Interview questions:**
  - [Medium Interview Questions](interview_questions/medium_questions.md)
- **Books:**

## **Plan**

**Objectives:**
- Fine-tune BERT for downstream tasks
- Apply BERT to text classification
- Learn task-specific fine-tuning strategies
- Understand learning rate scheduling for fine-tuning
- Evaluate fine-tuned BERT

**Deep Learning Concept(s):**
- Fine-tuning: adapting pre-trained model to specific task
- Task-specific heads (classification, QA, NER)
- Transfer learning with BERT
- Discriminative fine-tuning
- Task adaptation

**Tools Used:**
- Hugging Face: `transformers.BertForSequenceClassification`, `transformers.Trainer`, `transformers.AutoModelForSequenceClassification`
- Datasets: GLUE, sentiment analysis datasets
- Functions: `Trainer()`, fine-tuning workflow

**Key Learnings:**
- Sequence classification: `BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)`
- Fine-tuning: train all layers or freeze some, typically use lower learning rate (1e-5 to 5e-5)
- Using Trainer: `Trainer(model=model, args=training_args, train_dataset=dataset, ...)`
- Task-specific heads: add classification head on top of BERT
- Learning rate: much lower than training from scratch (pre-trained weights are good)
- Epochs: typically 3-5 epochs (fewer than training from scratch)
- Evaluation: fine-tuned BERT achieves SOTA on many NLP tasks
- Understanding that fine-tuning adapts general language knowledge to specific task

**References:**
- **Paper**: Devlin, J., et al. (2018). "BERT: Pre-training..." - Section 4
- **Tutorial**: [Hugging Face: Fine-tuning BERT](https://huggingface.co/docs/transformers/training)
- **Tutorial**: [Fine-tuning Tutorial](https://huggingface.co/docs/transformers/tasks/sequence_classification)
- **Dataset**: [GLUE Benchmark](https://gluebenchmark.com/)

---