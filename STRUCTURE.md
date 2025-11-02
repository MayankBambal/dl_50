# Repository Structure Guide

This document provides a detailed overview of the repository structure and how to navigate it.

## 📁 Complete Directory Structure

```
dl_50/
├── README.md                    # Main course overview and navigation
├── LICENSE                      # MIT License
├── CONTRIBUTING.md              # Contribution guidelines
├── STRUCTURE.md                 # This file
├── .gitignore                   # Git ignore rules
├── requirements.txt             # Python dependencies
├── plan.csv                     # Course plan with all 50 days
│
├── daily/                       # Main course content
│   ├── README.md               # Overview of daily structure
│   ├── day_template/           # Template for creating new days
│   │   ├── README.md
│   │   ├── notebooks/
│   │   ├── code/
│   │   ├── data/
│   │   └── outputs/
│   ├── day 1/                  # Day 1: What is Deep Learning?
│   ├── day 2/                  # Day 2: The Perceptron Explained
│   └── ... (through day 50)
│
├── docs/                        # Documentation
│   ├── syllabus.md             # Detailed course syllabus
│   ├── installation.md         # Setup instructions
│   ├── references.md           # Papers, books, resources
│   └── faq.md                  # Frequently asked questions
│
├── data/                        # Shared datasets
│   ├── raw/                    # Original, unprocessed data
│   ├── processed/              # Cleaned/preprocessed data
│   └── external/               # External datasets (references)
│
├── models/                      # Saved models
│   ├── checkpoints/            # Training checkpoints
│   ├── trained/              # Final trained models
│   └── pretrained/             # Pre-trained models
│
├── utils/                       # Utility functions
│   ├── __init__.py
│   ├── README.md
│   ├── data_loader.py          # Data loading utilities
│   ├── visualization.py        # Plotting functions
│   ├── metrics.py              # Evaluation metrics
│   └── helpers.py              # General helpers
│
├── scripts/                     # Helper scripts
│   ├── README.md
│   ├── setup_environment.sh    # Environment setup
│   └── download_data.sh        # Data download script
│
├── projects/                    # Larger projects
│   └── [Project folders]
│
├── assets/                      # Media files
│   ├── images/                 # Diagrams, screenshots
│   └── diagrams/               # Architecture diagrams
│
└── tests/                       # Unit tests
    └── [Test files]
```

## 📝 File Descriptions

### Root Files

- **README.md**: Course overview, prerequisites, daily navigation links
- **LICENSE**: MIT License (open source)
- **CONTRIBUTING.md**: Guidelines for contributing
- **STRUCTURE.md**: This file - repository structure guide
- **.gitignore**: Files/directories to ignore in git
- **requirements.txt**: Python package dependencies
- **plan.csv**: Course curriculum with all 50 days

### Daily Folders Structure

Each day folder (day 1 through day 50) contains:

```
day XX/
├── README.md              # Day overview, objectives, exercises
├── notebooks/             # Jupyter notebooks
│   └── day_XX_exercise.ipynb
├── code/                  # Python scripts (if any)
├── data/                  # Day-specific datasets
└── outputs/               # Generated plots, predictions
```

### Documentation (docs/)

- **syllabus.md**: Week-by-week breakdown, learning outcomes
- **installation.md**: Step-by-step setup guide
- **references.md**: Papers, books, online resources
- **faq.md**: Common questions and troubleshooting

### Data Directory

- **raw/**: Original datasets (usually not tracked in git)
- **processed/**: Preprocessed data ready for modeling
- **external/**: Links/references to external datasets

### Models Directory

- **checkpoints/**: Saved model weights during training
- **trained/**: Final trained models
- **pretrained/**: Pre-trained models for transfer learning

> **Note**: Large model files are typically not tracked in git (see .gitignore)

## 🔍 How to Navigate

### For New Learners

1. Start with [README.md](README.md) for overview
2. Follow [installation.md](docs/installation.md) for setup
3. Begin with [Day 1](daily/day%201/)
4. Check [FAQ](docs/faq.md) if you have questions

### For Contributors

1. Read [CONTRIBUTING.md](CONTRIBUTING.md)
2. Check existing code style
3. Follow the day_template structure
4. Submit pull requests

### For Course Creators

1. Use [day_template](daily/day_template/) as reference
2. Follow naming conventions (day XX)
3. Update main README.md when adding days
4. Keep structure consistent

## 📊 Data Flow

```
Raw Data → Processing → Model Training → Evaluation → Deployment
    ↓           ↓             ↓              ↓            ↓
data/raw/  data/processed/  models/     outputs/    projects/
```

## 🎯 Best Practices

1. **Consistent Structure**: All days follow the same folder structure
2. **Clear Naming**: Use descriptive names (day XX, not day1)
3. **Documentation**: Every folder has a README explaining its purpose
4. **Git Hygiene**: Use .gitignore to exclude large files
5. **Modular Code**: Keep utilities in utils/, reusable scripts in scripts/

## 🔗 Quick Links

- [Course Overview](README.md)
- [Day 1](daily/day%201/)
- [Installation Guide](docs/installation.md)
- [Syllabus](docs/syllabus.md)
- [References](docs/references.md)

---

**Tip**: Bookmark this file and refer to it when organizing new content!

