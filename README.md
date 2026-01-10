# 🧠 Handwritten Digit Recognizer (GUI)

A desktop application for recognizing handwritten digits using a Convolutional Neural Network (CNN) built with **PyTorch** and **Tkinter**.

The project features a custom drawing interface with **Smart Centering algorithm**, ensuring high accuracy even for off-center or small drawings.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)
![License](https://img.shields.io/badge/License-MIT-green)

## ✨ Features
- **Custom CNN Architecture**: Lightweight model (2 Conv layers, Dropout, FC layers).
- **GUI Interface**: Built with `tkinter` (no browser required).
- **Smart Preprocessing**: Automatically crops and centers the digit to match MNIST format (28x28).
- **Local Training**: Train the model on your CPU/GPU directly from the app.
- **Cross-Platform**: Works on Windows, macOS, and Linux.

## 🖼️Screenshots
![Train](Train.png)
![Recognize](Recognize.png)

## 🛠 Installation

### Prerequisites
You need **Python 3.10+** installed.

### 1. Clone the repository
```bash
git clone https://github.com/v0id-core/digit-recognizer.git
cd digit-recognizer
```

### 2. Set up a Virtual Environment (Recommended)
Windows:
```bash
python -m venv venv
venv\Scripts\activate
```

macOS / Linux:
```bash
python3 -m venv venv
source venv/bin/activate

```
### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the desktop application
```bash
python desktop_app.py
```

## 🚀 Usage
- Train the model: If you launch the app for the first time, click "Train Model" in the app interface. The app will download the MNIST dataset and train the neural network (~1-2 minutes).
- Draw: Use your mouse to draw a digit (0-9) on the canvas.
- Recognize: Click "Recognize" to see the prediction and confidence score.

## 🤖 Tech Stack
- Core: Python 3.10+
- ML Framework: PyTorch, Torchvision
- GUI: Tkinter
- Image Processing: PIL (Pillow), NumPy

## ⚙️ Troubleshooting
- App doesn't start? Ensure that you have Python 3.10+ and all dependencies installed correctly. Check version: python --version.
- Training fails? Check your internet connection. The MNIST dataset requires downloading when you press "Train Model".

## 📝 License
- This project is licensed under the MIT License. You can freely use, modify, and distribute the code.

## 👨‍💻 Author
- v0id-core
- GitHub: https://github.com/v0id-core
Usage