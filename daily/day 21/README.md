## **To Do**

**Level 1 (Product Manager Study):**
- **Video:**
- **Blog:**

**Level 2 (Junior Data Scientist):**
- **Code:**
  - [day21.ipynb](notebooks/day21.ipynb)
- **Books:**
- **Interview questions:**
  - [Easy Interview Questions](interview_questions/easy_questions.md)

**Level 3 (Data Scientist):**
- **Interview questions:**
  - [Medium Interview Questions](interview_questions/medium_questions.md)
- **Books:**

## **Plan**

**Objectives:**
- Apply transfer learning to a real problem
- Use VGG or ResNet architecture
- Train on custom dataset
- Implement data augmentation
- Compare fine-tuning vs. feature extraction

**Deep Learning Concept(s):**
- VGG architecture: deep network with small filters (3x3)
- ResNet architecture: residual connections, skip connections
- Data augmentation for better generalization
- Custom dataset creation
- Evaluation metrics for image classification

**Tools Used:**
- torchvision: `torchvision.models.vgg16()`, `torchvision.models.resnet50()`
- torchvision transforms: `transforms.Compose()`, `transforms.RandomHorizontalFlip()`, `transforms.RandomRotation()`, `transforms.ColorJitter()`
- Custom datasets: `torch.utils.data.Dataset`
- Functions: Building custom dataset classes, applying augmentation

**Key Learnings:**
- VGG architecture: `model = torchvision.models.vgg16(weights='IMAGENET1K_V1')`
- ResNet architecture: `model = torchvision.models.resnet50(weights='IMAGENET1K_V1')`
- Data augmentation: `transforms.Compose([transforms.RandomHorizontalFlip(), transforms.RandomRotation(10), transforms.ToTensor()])`
- Creating custom dataset: `class CustomDataset(Dataset): def __init__(), def __len__(), def __getitem__()`
- Fine-tuning strategies:
  - Option 1: Train only classifier with frozen backbone
  - Option 2: Fine-tune last few layers
  - Option 3: Fine-tune all layers with discriminative learning rates
- Understanding residual connections: identity mapping + learned residual
- Using different pre-trained models for different tasks
- Evaluating on custom test set

**References:**
- **Paper**: Simonyan, K., & Zisserman, A. (2014). "Very Deep Convolutional Networks for Large-Scale Image Recognition" (VGG)
- **Paper**: He, K., et al. (2016). "Deep Residual Learning for Image Recognition" (ResNet)
- **Tutorial**: [PyTorch: Transfer Learning Tutorial](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)
- **Tutorial**: [Custom Datasets in PyTorch](https://pytorch.org/tutorials/beginner/data_loading_tutorial.html)
- **Online**: [Understanding ResNet](https://towardsdatascience.com/understanding-and-visualizing-resnets-442d831b41b8)

---