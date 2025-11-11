## **To Do**

**Level 1 (Product Manager Study):**
- **Video:**
- **Blog:**

**Level 2 (Junior Data Scientist):**
- **Code:**
  - [day20.ipynb](notebooks/day20.ipynb)
- **Books:**
- **Interview questions:**
  - [Easy Interview Questions](interview_questions/easy_questions.md)

**Level 3 (Data Scientist):**
- **Interview questions:**
  - [Medium Interview Questions](interview_questions/medium_questions.md)
- **Books:**

## **Plan**

**Objectives:**
- Understand what transfer learning is and why it's powerful
- Learn to use pre-trained models
- Fine-tune pre-trained models for new tasks
- Understand feature extraction vs. fine-tuning
- Load pre-trained models from torchvision

**Deep Learning Concept(s):**
- Transfer learning definition
- Pre-trained models and their benefits
- Feature extraction: freeze backbone, train classifier
- Fine-tuning: update all or some layers
- Domain adaptation
- Why ImageNet pre-training works

**Tools Used:**
- torchvision models: `torchvision.models.resnet18()`, `torchvision.models.vgg16()`, `torchvision.models.efficientnet_b0()`
- Functions: `models.resnet18(pretrained=True)`, freezing layers: `param.requires_grad = False`, modifying classifier head

**Key Learnings:**
- Loading pre-trained model: `model = torchvision.models.resnet18(pretrained=True)` or `weights='IMAGENET1K_V1'`
- Freezing layers: `for param in model.parameters(): param.requires_grad = False`
- Modifying classifier: `model.fc = nn.Linear(model.fc.in_features, num_classes)` for ResNet
- Two transfer learning approaches:
  - Feature extraction: freeze backbone, only train classifier
  - Fine-tuning: train all layers (or last few) with lower learning rate
- Learning rate scheduling for fine-tuning: typically lower LR for pre-trained parts
- Understanding that pre-trained models learned general features
- Using different architectures: ResNet, VGG, EfficientNet, etc.

**References:**
- **Book**: "Hands-On Machine Learning" by Aurélien Géron - Chapter 14.6 (Transfer Learning)
- **Tutorial**: [PyTorch: Transfer Learning Tutorial](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)
- **Tutorial**: [torchvision Models](https://pytorch.org/vision/stable/models.html)
- **Online**: [Transfer Learning Guide](https://towardsdatascience.com/transfer-learning-from-pre-trained-models-f2393f124751)

---