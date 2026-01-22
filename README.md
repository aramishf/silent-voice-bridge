---
title: Silent Voice Bridge
emoji: 🤟
colorFrom: purple
colorTo: pink
sdk: docker
pinned: false
---

# Silent Voice Bridge 🤟

**Real-time ASL Fingerspelling Recognition & Sentence Formation System**

A complete machine learning solution that translates American Sign Language (ASL) fingerspelling into text in real-time, enabling seamless communication and practice for ASL learners and users.

---

## 🎯 Overview

Silent Voice Bridge is an intelligent ASL recognition system that uses computer vision and deep learning to recognize hand gestures for all 26 letters (A-Z) and 10 digits (0-9), allowing users to spell out words and form complete sentences through sign language.

### Key Features

✨ **Real-time Recognition** - Instant detection and translation of ASL fingerspelling at 30 FPS  
🧠 **Smart Sentence Formation** - Automatically builds words and sentences from detected letters  
🔄 **Intelligent Deduplication** - Prevents repeated letters (H-H-H → H) with release detection  
🎯 **Context-Aware Disambiguation** - Automatically converts similar letters/digits based on context (O/0, V/2, B/4, etc.)  
✏️ **Auto-Correction** - Fixes common spelling mistakes (HELPO → HELLO)  
📊 **Split-Screen Interface** - Live video feed with real-time transcription panel  
📝 **Sentence History** - Saves completed sentences with timestamps  
🎓 **Perfect for Learning** - Ideal for ASL practice and skill development  

---

## 🚀 Use Cases

### For ASL Learners
- **Practice fingerspelling** with instant feedback
- **Build vocabulary** by spelling words letter-by-letter
- **Track progress** with sentence history
- **Learn at your own pace** with pause functionality

### For Communication
- **Spell out words** that don't have dedicated signs
- **Form complete sentences** for complex communication
- **Bridge language gaps** between ASL users and non-signers
- **Create text transcripts** of fingerspelled conversations

### For Education
- **Teaching tool** for ASL instructors
- **Self-study aid** for students learning ASL
- **Assessment tool** to evaluate fingerspelling accuracy
- **Interactive practice** sessions

---

## 🏗️ System Architecture

### Technology Stack
- **Computer Vision**: MediaPipe Hands for landmark detection
- **Deep Learning**: Bidirectional LSTM neural network (PyTorch)
- **Feature Extraction**: 258-dimensional hand landmark vectors
- **Real-time Processing**: OpenCV for video capture and display

### Model Performance
- **Validation Accuracy**: 100%
- **Test Accuracy**: 99.62%
- **Dataset Size**: 7,000+ training samples
- **Recognition Speed**: Real-time (30 FPS)
- **Supported Classes**: 36 (A-Z, 0-9)

### Core Components

1. **Hand Tracking** - MediaPipe extracts 21 hand landmarks per frame
2. **Feature Extraction** - Converts landmarks to 258-feature vectors
3. **Sequence Processing** - LSTM analyzes 30-frame sequences (1 second)
4. **Sentence Builder** - Intelligent text formation with context awareness
5. **User Interface** - Split-screen display with video and transcription

---

## 💡 How It Works

### Recognition Pipeline

```
Hand Gesture → MediaPipe Detection → Feature Extraction → LSTM Model → Letter Prediction
                                                                              ↓
Sentence History ← Completed Sentence ← Word Formation ← Smart Deduplication
```

### Intelligent Features

**Deduplication with Release Detection**
- Detects when you hold a letter (e.g., L in "HELLO")
- Shows "Release hand to add another 'L'" message
- Adds second L when you briefly release and re-hold

**Context-Aware Disambiguation**
- Spelling "HELLO"? Converts 0 → O automatically
- Typing "2024"? Keeps digits as digits
- Handles similar pairs: O/0, V/2, W/6, B/4, D/1, F/9

**Auto-Correction**
- Automatically fixes common mistakes
- HELPO → HELLO, WROLD → WORLD, THNAK → THANK

---

## 🐳 Docker Deployment (Recommended)

**One-command deployment** - No manual installation required!

### Prerequisites
1. Install [Docker Desktop](https://www.docker.com/products/docker-desktop/)
2. (Mac only) Install [XQuartz](https://www.xquartz.org/) for GUI support

### Quick Start
```bash
./docker-start.sh
```

That's it! The container will:
- ✅ Build automatically with all dependencies
- ✅ Access your webcam
- ✅ Display the GUI
- ✅ Persist your trained models

### Docker Commands
```bash
# Start
docker-compose up -d

# View logs
docker-compose logs -f

# Stop
docker-compose down

# Train inside container
docker-compose exec asl-inference python src/train.py
```

See [DOCKER.md](DOCKER.md) for detailed instructions.

---

## 🎮 Usage (Local Installation)

### Quick Start

```bash
# 1. Collect training data (optional - pre-trained model included)
./start_bulk_collection.sh

# 2. Train model (optional)
cd src
../venv/bin/python preprocessing.py
../venv/bin/python train.py

# 3. Run sentence formation system
../venv/bin/python sentence_inference.py --model ../models/checkpoints/best_model.pth
```

### Controls

**During Sentence Formation:**
- `SPACE` - Add space between words
- `BACKSPACE` - Delete last character
- `ENTER` - Finish sentence (save to history)
- `C` - Clear current sentence
- `P` - Pause/Resume
- `Q` - Quit

### Example Workflow

1. **Start the system** - Webcam opens with split-screen interface
2. **Make ASL gestures** - Spell out letters one by one
3. **Watch auto-formation** - System builds words with smart deduplication
4. **Add spaces** - Press SPACE between words
5. **Complete sentences** - Press ENTER to save to history
6. **Review history** - See all your completed sentences with timestamps

---

## 📊 Training Your Own Model

### Data Collection (Fast Mode)

The bulk collection system captures ~6 samples per second:

```bash
./start_bulk_collection.sh
```

**Tips for Quality Data:**
- Hold gesture clearly and move hand around
- Vary angles, distances, and positions
- Collect 150-200 samples per class
- For motion letters (J, Z), perform the motion repeatedly

**Time Estimate:** 15-30 minutes for all 36 classes

### Model Training

```bash
cd src
../venv/bin/python preprocessing.py  # Normalize and split data
../venv/bin/python train.py          # Train LSTM model
```

**Training Time:** ~3-5 minutes on CPU

---

## 🎓 Educational Value

### For Students
- **Interactive learning** - Immediate feedback on fingerspelling
- **Self-paced practice** - No instructor needed
- **Progress tracking** - Review sentence history
- **Confidence building** - Practice without judgment

### For Instructors
- **Teaching aid** - Demonstrate correct fingerspelling
- **Assessment tool** - Evaluate student accuracy
- **Homework assignments** - Students practice at home
- **Progress monitoring** - Track improvement over time

### Learning Outcomes
- Master ASL alphabet (A-Z)
- Learn ASL digits (0-9)
- Improve fingerspelling speed and accuracy
- Build vocabulary through spelling practice
- Develop muscle memory for hand shapes

---

## 🔬 Technical Details

### Model Architecture
- **Type**: Bidirectional LSTM
- **Layers**: 2 LSTM layers (128 → 64 hidden units)
- **Input**: 30 frames × 258 features
- **Output**: 36 classes (A-Z, 0-9)
- **Parameters**: 566,820 trainable parameters

### Feature Engineering
- **Hand Landmarks**: 21 points per hand (x, y, z coordinates)
- **Temporal Context**: 30-frame sequences (1 second at 30 FPS)
- **Normalization**: StandardScaler for feature scaling
- **Data Augmentation**: Natural variation through hand movement

### Performance Metrics
- **Precision**: 99-100% across all classes
- **Recall**: 96-100% across all classes
- **F1-Score**: 98-100% across all classes
- **Inference Time**: ~10ms per prediction

---

## 🌟 Future Enhancements

- [ ] Support for full ASL words and phrases
- [ ] Multi-hand gesture recognition
- [ ] Mobile app deployment (iOS/Android)
- [ ] Web-based interface
- [ ] Voice synthesis for audio output
- [ ] Integration with communication apps
- [ ] Expanded vocabulary with common phrases
- [ ] Multi-language support

---

## 🤝 Contributing

This project welcomes contributions! Areas for improvement:
- Additional ASL gestures and signs
- Enhanced auto-correction dictionary
- UI/UX improvements
- Performance optimizations
- Documentation and tutorials

---

## 📄 License

MIT License - Feel free to use for educational and personal projects

---

## 🙏 Acknowledgments

- **MediaPipe** - Hand tracking technology
- **PyTorch** - Deep learning framework
- **ASL Community** - Inspiration and guidance
- **Open Source Community** - Tools and libraries

---

## 📞 Support

For questions, issues, or suggestions:
- Open an issue on GitHub
- Check the documentation in `/docs`
- Review example videos in `/examples`

---

**Built with ❤️ for the Deaf and Hard of Hearing community and ASL learners worldwide**

---

## 📸 Screenshots

### Split-Screen Interface
- Left: Live video feed with hand tracking
- Right: Current sentence, detection status, and history

### Features in Action
- Real-time letter detection with confidence scores
- Automatic word formation with smart deduplication
- Context-aware disambiguation (O/0, V/2, etc.)
- Sentence history with timestamps

---

## 🎯 Quick Stats

- **36 Classes**: A-Z letters + 0-9 digits
- **99.62% Accuracy**: State-of-the-art performance
- **Real-time**: 30 FPS processing
- **7,000+ Samples**: Comprehensive training dataset
- **Smart Features**: Deduplication, disambiguation, auto-correction
- **User-Friendly**: Intuitive split-screen interface

---

**Start bridging the communication gap with Silent Voice Bridge today!** 🌉
