## **To Do**

**Level 1 (Product Manager Study):**
- **Video:**
- **Blog:**

**Level 2 (Junior Data Scientist):**
- **Code:**
  - [day47.ipynb](notebooks/day47.ipynb)
- **Books:**
- **Interview questions:**
  - [Easy Interview Questions](interview_questions/easy_questions.md)

**Level 3 (Data Scientist):**
- **Interview questions:**
  - [Medium Interview Questions](interview_questions/medium_questions.md)
- **Books:**

## **Plan**

**Objectives:**
- Understand causal language modeling objective
- Learn how GPT generates text
- Implement text generation with GPT
- Understand autoregressive generation
- Learn sampling strategies (greedy, top-k, nucleus)

**Deep Learning Concept(s):**
- Causal language modeling: P(token_t | tokens_<t)
- Autoregressive generation: generating one token at a time
- Generation strategies: greedy, sampling, top-k, nucleus (top-p)
- Temperature for controlling randomness
- Prompting: providing context for generation

**Tools Used:**
- Hugging Face: `transformers.GPT2LMHeadModel`, `transformers.GPT2Tokenizer`
- Generation: `model.generate()` function
- Functions: text generation, sampling strategies

**Key Learnings:**
- Generation: `model.generate(input_ids, max_length=100, num_return_sequences=1)`
- Greedy: `do_sample=False` (always picks highest probability token)
- Sampling: `do_sample=True, temperature=1.0`
- Top-k sampling: `top_k=50` (sample from top k tokens)
- Nucleus sampling: `top_p=0.9` (sample from tokens covering 90% probability mass)
- Autoregressive: each generated token becomes input for next step
- Prompting: providing initial text, model continues from there
- Understanding that GPT learns from next-token prediction during training

**References:**
- **Paper**: Radford, A., et al. (2019). "Language Models are Unsupervised Multitask Learners"
- **Tutorial**: [Hugging Face: Text Generation](https://huggingface.co/docs/transformers/tasks/language_modeling)
- **Tutorial**: [Text Generation with GPT-2](https://huggingface.co/blog/how-to-generate)
- **Online**: [Understanding Text Generation](https://towardsdatascience.com/how-to-sample-from-language-models-682bceb97277)

---