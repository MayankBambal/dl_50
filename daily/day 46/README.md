## **To Do**

**Level 1 (Product Manager Study):**
- **Video:**
- **Blog:**

**Level 2 (Junior Data Scientist):**
- **Code:**
  - [day46.ipynb](notebooks/day46.ipynb)
- **Books:**
- **Interview questions:**
  - [Easy Interview Questions](interview_questions/easy_questions.md)

**Level 3 (Data Scientist):**
- **Interview questions:**
  - [Medium Interview Questions](interview_questions/medium_questions.md)
- **Books:**

## **Plan**

**Objectives:**
- Understand GPT as decoder-only Transformer
- Learn GPT architecture details
- Understand causal (masked) self-attention
- Learn about GPT variants (GPT-1, GPT-2, GPT-3)
- Compare GPT vs. BERT

**Deep Learning Concept(s):**
- GPT: Generative Pre-trained Transformer
- Decoder-only architecture (no encoder)
- Causal/autoregressive self-attention
- Language modeling objective
- Generation capabilities

**Tools Used:**
- Hugging Face: `transformers.GPT2Model`, `transformers.GPT2LMHeadModel`
- Functions: `GPT2LMHeadModel.from_pretrained()`, understanding architecture

**Key Learnings:**
- Loading GPT: `GPT2LMHeadModel.from_pretrained('gpt2')`
- GPT architecture: only Transformer decoder (masked self-attention, no encoder-decoder attention)
- Causal masking: can only attend to previous tokens (autoregressive)
- GPT vs. BERT: GPT is generative (predicts next token), BERT is bidirectional
- GPT variants: GPT-1 (117M), GPT-2 (1.5B), GPT-3 (175B parameters)
- Understanding that GPT generates text token by token
- Training: predicting next token given previous tokens (causal LM)
- Tokenizer: BPE (Byte Pair Encoding)

**References:**
- **Paper**: Radford, A., et al. (2018). "Improving Language Understanding by Generative Pre-Training" (GPT-1)
- **Paper**: Radford, A., et al. (2019). "Language Models are Unsupervised Multitask Learners" (GPT-2)
- **Tutorial**: [Hugging Face: GPT-2](https://huggingface.co/docs/transformers/model_doc/gpt2)
- **Visual Guide**: [GPT Explained](https://jalammar.github.io/how-gpt3-works-visualizations-animations/)

---