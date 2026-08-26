# 🖼️ AI Image Enhancer API

A FastAPI service that upscales and enhances images by **4x** using [RealESRGAN](https://github.com/xinntao/Real-ESRGAN). Upload a blurry or small image, get back a crisp, high-resolution version.

## ✨ Features

- 🚀 **4x upscaling** with the RealESRGAN `x4plus` model
- 📤 Simple `multipart/form-data` upload endpoint
- 🖼️ Returns the enhanced image as a downloadable file
- ☁️ One-click deploy to Render via included `render.yaml`
- 🧠 Auto-downloads model weights on first boot

## 🛠️ Tech Stack

- **Python** · **FastAPI** · **PyTorch** · **RealESRGAN** · **Pillow**

## 🚀 Getting Started

```bash
# 1. Clone the repo
git clone https://github.com/Sugamdeol/ai-enhacer-api.git
cd ai-enhacer-api

# 2. Create a virtualenv and install dependencies
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 3. Run the server
uvicorn app:app --host 0.0.0.0 --port 8000
```

Model weights (~64 MB) are downloaded automatically to `weights/` on first request.

## 📡 API Usage

```bash
curl -X POST http://localhost:8000/enhance \
  -F "file=@photo.jpg" \
  -o enhanced.png
```

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

Built with ❤️ by [Sugam Deol](https://github.com/Sugamdeol)
