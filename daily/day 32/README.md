## **To Do**

**Level 1 (Product Manager Study):**
- **Video:**
- **Blog:**

**Level 2 (Junior Data Scientist):**
- **Code:**
  - [day32.ipynb](notebooks/day32.ipynb)
- **Books:**
- **Interview questions:**
  - [Easy Interview Questions](interview_questions/easy_questions.md)

**Level 3 (Data Scientist):**
- **Interview questions:**
  - [Medium Interview Questions](interview_questions/medium_questions.md)
- **Books:**

## **Plan**

**Objectives:**
- Understand the bottleneck problem in Seq2Seq
- Learn why fixed-size context vector is limiting
- Understand information compression issues
- Recognize long sequence handling problems
- See why attention is needed

**Deep Learning Concept(s):**
- Bottleneck problem: compressing all information into fixed-size vector
- Information loss in long sequences
- Fixed context vector limitations
- Difficulty with long-range dependencies
- Why we need dynamic context

**Tools Used:**
- Analysis and visualization of encoder-decoder limitations
- Comparison of different sequence lengths

**Key Learnings:**
- Understanding that fixed-size context loses information
- Long sequences: hard to compress into single vector
- Each position in output attends to same context (not ideal)
- Motivation for attention: different output positions need different parts of input
- Understanding that attention will solve these problems
- Visualizing information bottleneck
- Preparing for attention mechanism (next day)

**References:**
- **Paper**: Bahdanau, D., et al. (2014). "Neural Machine Translation by Jointly Learning to Align and Translate"
- **Book**: "Deep Learning" by Ian Goodfellow - Chapter 10.4 (Limitations)
- **Online**: [The Bottleneck Problem in Seq2Seq](https://towardsdatascience.com/attention-in-neural-networks-e66920838742)

---