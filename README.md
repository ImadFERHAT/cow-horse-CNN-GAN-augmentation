# Cow vs Horse Classification with GAN-based Augmentation

## 📌 Project Overview  
This project implements a **Convolutional Neural Network (CNN)** to classify images of **cows** and **horses**.  
To improve classification accuracy, the dataset is **augmented using generative models**:  
- **GAN** (Generative Adversarial Network)  
- **cGAN** (Conditional GAN)  
- **CycleGAN** (Unpaired Image-to-Image Translation)  

By increasing dataset diversity and size, the classifier becomes more robust and generalizes better.

---

## 🎯 Objectives  
1. Build a **baseline CNN classifier** for cows vs horses.  
2. Perform **dataset augmentation** with:
   - GAN (unconditional generation)
   - cGAN (class-conditional generation)
   - CycleGAN (style transfer between cows and horses)  
3. Compare model performance **before and after augmentation**.  

---

## 📂 Dataset  
- **Original dataset**: Images of cows and horses.  
- **Augmentation**:
  - **GAN & cGAN**: Generate synthetic animal images.
  - **CycleGAN**: Transfer style between classes (cow ↔ horse).  


---

## 🧠 Model Architectures  
### 1️⃣ CNN Classifier
- Input: 128×128 RGB images  
- Conv + Conv + MaxPooling + Linear 

### 2️⃣ Generative Models
- **GAN**: Basic DCGAN-style generator/discriminator.
- **cGAN**: Generator conditioned on class labels.
- **CycleGAN**: Two generators + two discriminators for style translation.

---

## ⚙️ Installation & Usage  
```bash
# Clone this repository
git clone https://github.com/ImadFERHAT/cow-horse-CNN-GAN-augmentation.git
cd cow-horse-CNN-GAN-augmentation

# Create virtual environment (optional)
python -m venv venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate      # Windows

# Install dependencies
pip install -r requirements.txt
