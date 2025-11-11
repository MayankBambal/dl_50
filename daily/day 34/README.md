## **To Do**

**Level 1 (Product Manager Study):**
- **Video:**
- **Blog:**

**Level 2 (Junior Data Scientist):**
- **Code:**
  - [day34.ipynb](notebooks/day34.ipynb)
- **Books:**
- **Interview questions:**
  - [Easy Interview Questions](interview_questions/easy_questions.md)

**Level 3 (Data Scientist):**
- **Interview questions:**
  - [Medium Interview Questions](interview_questions/medium_questions.md)
- **Books:**

## **Plan**

**Objectives:**
- Visualize attention weights
- Understand what attention patterns mean
- Create attention heatmaps
- Analyze attention in translation tasks
- Interpret model behavior through attention

**Deep Learning Concept(s):**
- Attention visualization techniques
- Attention heatmaps
- Interpreting attention patterns
- Alignment visualization
- Understanding what model focuses on

**Tools Used:**
- Matplotlib: `matplotlib.pyplot.imshow()` for heatmaps
- Seaborn: `seaborn.heatmap()` for better visualizations
- Functions: plotting attention weights as matrices

**Key Learnings:**
- Creating attention heatmaps: `plt.imshow(attention_weights, cmap='Blues')`
- Understanding attention patterns: diagonal = good alignment
- Interpreting attention: which input words map to which output words
- Attention visualization: rows = decoder positions, columns = encoder positions
- Using attention to debug models
- Understanding attention in different tasks
- Attention as interpretability tool

**References:**
- **Tutorial**: [Visualizing Attention](https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html)
- **Online**: [Attention Visualization Tools](https://github.com/jessevig/bertviz)
- **Paper**: Attention visualization examples from Bahdanau et al.

---