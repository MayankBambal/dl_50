# Frequently Asked Questions (FAQ)

Common questions and answers about the 50 Days of Deep Learning course.

## General Questions

### Q: What is the completion deadline?

**A:** The course will finish on **December 31st, 2025**. This is your 2025 resolution deadline. Plan your schedule to complete all 50 days by this date. If you start later, you can do multiple days per day or extend your daily time commitment to finish on time.

### Q: Do I need prior experience in machine learning?

**A:** No! This course is designed for beginners. However, basic Python programming knowledge and familiarity with NumPy/Pandas is recommended.

### Q: How long should each day take?

**A:** Most days take 1-2 hours, depending on your background and how thoroughly you work through exercises. Don't rush - understanding is more important than speed.

### Q: Can I skip days or do them out of order?

**A:** The course is designed to be sequential, with each day building on previous concepts. We recommend following the order, but you can adjust based on your needs.

### Q: Do I need a GPU?

**A:** No, but it's recommended for later weeks (especially weeks 3-7). Early weeks can be completed on CPU. GPU will significantly speed up training.

### Q: Which framework does this course use?

**A:** This course uses **PyTorch 2.5+** exclusively. We focus solely on PyTorch to maximize learning efficiency in this intensive 50-day format. All exercises, examples, and projects use PyTorch.

## Technical Questions

### Q: Installation is failing. What should I do?

**A:** 
1. Make sure you're using Python 3.12+ (latest stable version)
2. Create a fresh virtual environment
3. Upgrade pip: `pip install --upgrade pip`
4. Install PyTorch first: Visit [pytorch.org](https://pytorch.org/get-started/locally/) for the latest installation command
5. Then install other requirements: `pip install -r requirements.txt`
6. Install requirements one by one if batch installation fails
7. Check [Installation Guide](installation.md) for more help

### Q: My model is training too slowly.

**A:** 
- Reduce batch size
- Use a smaller dataset for practice
- Consider using GPU
- In later weeks, use pre-trained models (transfer learning)
- For exercises, you can use fewer epochs

### Q: I'm getting memory errors (OOM).

**A:**
- Reduce batch size
- Use gradient accumulation
- Work with smaller datasets
- Use CPU if GPU memory is limited
- Close other applications

### Q: How do I know if my model is correct?

**A:**
- Check loss is decreasing
- Validate on test set (not training set)
- Compare with expected results in notebooks
- Use provided test cases if available

### Q: What if I get stuck on an exercise?

**A:**
- Review the day's README
- Check previous days for similar concepts
- Search for the concept online (many tutorials available)
- Take a break and come back fresh
- Check GitHub issues for similar problems

## Content Questions

### Q: Are there solutions provided?

**A:** Solutions may be available in a separate branch or will be added over time. We encourage you to try solving exercises yourself first.

### Q: Can I use different datasets?

**A:** Absolutely! Once you understand the concepts, try applying them to your own data. This deepens understanding.

### Q: What if I finish early or want more practice?

**A:** Great! Try:
- Implementing variations of the exercises
- Applying concepts to your own projects
- Reading the referenced papers
- Contributing improvements to the course

### Q: What should I do after completing the course?

**A:**
- Build your own deep learning projects
- Explore specialized topics (GANs, Reinforcement Learning, etc.)
- Contribute to open-source ML projects
- Follow latest research papers
- Consider advanced courses or specializations
- Celebrate completing your 2025 resolution! 🎉

### Q: How can I stay motivated to finish by December 31st, 2025?

**A:** Here are proven strategies:
- **Start today** - Don't wait for the "perfect time"
- **Track your progress** - Mark off completed days in the main README
- **Share your commitment** - Tell others about your goal (accountability helps!)
- **Use #50DaysDeepLearning** - Share milestones on social media
- **Remember your why** - Visualize completing your 2025 resolution
- **Consistency over intensity** - Better to do 1 hour daily than 7 hours once a week
- **If you miss a day** - Don't give up! Just pick up the next day. The deadline is December 31st—you have time to catch up.

## Course Structure

### Q: Why 50 days?

**A:** 50 days provides a comprehensive journey while being achievable. You can adjust the pace to fit your schedule.

### Q: What's the difference between days and weeks?

**A:** Days are individual lessons. Weeks group related topics into themes. You'll progress day by day through the weekly themes.

### Q: Are there assignments or exams?

**A:** Each day has exercises, and there are weekly projects. The final day (Day 50) includes a capstone project. No formal exams - focus on learning!

## Git & Repository

### Q: Should I fork or clone this repository?

**A:** 
- **Clone** if you just want to follow along
- **Fork** if you want to customize and keep your own version

### Q: Can I contribute improvements?

**A:** Yes! Check [CONTRIBUTING.md](../CONTRIBUTING.md) for guidelines. Contributions are welcome.

### Q: How do I track my progress?

**A:** 
- Check off days in the main README.md
- Create your own progress.md file
- Use GitHub's project board
- Keep notes in each day's folder

## Hardware & Setup

### Q: Can I run this on Windows/Mac/Linux?

**A:** Yes! All operating systems are supported. See [Installation Guide](installation.md) for OS-specific instructions.

### Q: Do I need to install everything at once?

**A:** No. You can install packages as needed. However, installing all at once (via requirements.txt) is more convenient.

### Q: Can I use Google Colab instead of local setup?

**A:** Yes! Colab is great for this course. You can upload notebooks to Colab. Note: You'll need to upload data files too.

## Learning Strategy

### Q: Should I take notes?

**A:** Highly recommended! Taking notes helps retention. Consider:
- Writing summaries of key concepts
- Creating your own code examples
- Drawing diagrams of architectures

### Q: How do I know if I understand a concept?

**A:** You understand it when you can:
- Explain it in your own words
- Implement it without looking at examples
- Identify when to use it
- Debug issues related to it

### Q: Should I memorize everything?

**A:** No! Focus on understanding concepts. You can always reference documentation. What's important is knowing *what* exists and *when* to use it.

### Q: What if I fall behind?

**A:** Don't worry! The course is self-paced. Focus on understanding rather than rushing. It's better to take longer and learn properly than rush through.

## Community & Support

### Q: Is there a community or forum?

**A:** Check the repository's Discussions or Issues section. You can also join relevant ML communities (see [References](references.md)).

### Q: How do I report bugs?

**A:** Open an issue on GitHub with:
- Description of the problem
- Steps to reproduce
- Error messages
- Your system info

### Q: Can I share my solutions?

**A:** Yes, but be mindful of spoilers. Consider using spoiler tags or sharing in a solutions branch.

---

**Have a question not listed here?** Open an issue on GitHub or check the [Installation Guide](installation.md) and [References](references.md).

