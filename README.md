# 🕵️‍♂️ Foto Forensics

**Foto Forensics** is a web application designed for **advanced digital image analysis**, focusing on **forensic authenticity verification, device identification (PRNU)**, and **steganography detection** using **machine learning**.

The project combines elements of **digital forensics, data security, and computer vision** to create a powerful analytical platform for professionals.  
This application is part of an **engineering thesis project**, accompanied by a theoretical and research-based diploma paper on the same topic.

![Start page](https://github.com/JakubTeichman/foto-forensics/blob/main/backend/static/image.png)

---

## 🚀 Features

- 🔍 **Metadata analysis** (EXIF, GPS, camera info)
- 🧠 **Machine learning–based steganography detection**
- 🖼️ **Device fingerprinting (PRNU analysis)**
- 📊 **Dynamic forensic report visualization**
- 🌙 **Modern dark-mode user interface**
- ⚙️ **Dockerized full-stack environment**


---

## 🐳 Run with Docker

The project is fully containerized using **Docker** for simple and consistent deployment.

### 1️⃣ Clone the repository

```bash
git clone https://github.com/your-repo/foto-forensics.git
cd foto-forensics
```

### 2️⃣ Build containers

```bash
docker-compose build
```

### 3️⃣ Start the application

```bash
docker-compose build
```

Once running, the app will be available at:
  👉 http://localhost:3000

---

## 📁 Project Structure

Below is a simplified overview of the project structure:

```bash
└── foto-forensics/
    ├── backend/
    │   ├── noiseprint/
    │   ├── prnu_utils/
    │   ├── routes/
    │   ├── static/
    │   ├── steganalysis/
    │   ├── stegano_compare/
    │   ├── app.py
    │   └── dockerfile
    ├── frontend/
    │   ├── src/
    │   │   ├── components/
    │   │   ├── App.tsx
    │   │   └── index.tsx
    │   └── dockerfile
    ├── models_trening/
    └── docker-compose.yml
```

## 🧠 About the Project

**Foto Forensics** is a full-stack web application created as part of an **engineering thesis project** in the field of *Modern Technologies in Forensics*.  
The system provides a digital forensics environment that allows users to analyze images for:

- 🔍 **Hidden or manipulated content detection (Steganography)**
- 🧠 **Device identification**
- 🧾 **EXIF and metadata extraction**
- 📊 **Visualization of forensic results through interactive reports**

The goal of this project is to make advanced forensic image analysis accessible through an intuitive and visually engaging web interface.  
The project is accompanied by a diploma thesis that expands on the theoretical and technical aspects of the system.

---

## ⚙️ Technologies Used

### 🖥️ Frontend
- **React + TypeScript** – component-based UI framework  
- **Tailwind CSS** – modern utility-first styling  
- **ECharts** – interactive data visualization 

### 🔧 Backend
- **Flask (Python)** – lightweight REST API framework  
- **OpenCV**, **NumPy**, **SciPy** – image and signal processing

### 🐳 Deployment
- **Docker & Docker Compose** – isolated, reproducible environment 

---

## 👨‍🎓 Author

**Jakub Teichman**  
Engineering Thesis Project – *„A System for Device Identification and Hidden Content Analysis in Images Based on Digital Forensics Techniques”*  
AGH, 2025  

📚 Focus Areas:
- Machine Learning for Digital Forensics  
- Image Integrity Verification  
- Steganography and Device Fingerprinting  

---

## 📄 License

This project is licensed under the **Creative Commons BY-NC 4.0**  
(CC BY-NC 4.0 – Attribution–NonCommercial).

This means that you are free to:
- copy, modify, and share the project,
- as long as you give appropriate credit,
- and do **not** use the material for commercial purposes.

Full license text:  
https://creativecommons.org/licenses/by-nc/4.0/

---

## ✉️ Contact

For questions, collaboration opportunities, or technical inquiries:

- 📧 **jakub.teichman@onet.pl**

---

*Foto Forensics – Analyze. Verify. Trust.*





