import io
import os
import requests
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse, JSONResponse
from PIL import Image
import torch
from realesrgan import RealESRGAN

app = FastAPI()

MODEL_PATH = "weights/RealESRGAN_x4plus.pth"
MODEL_URL = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"

device = torch.device("cpu")
model = None


def download_model():
    os.makedirs("weights", exist_ok=True)
    if not os.path.exists(MODEL_PATH):
        print("Downloading model...")
        r = requests.get(MODEL_URL, stream=True)
        with open(MODEL_PATH, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
        print("Model downloaded")


def load_model():
    global model
    download_model()
    model = RealESRGAN(device, scale=4)
    model.load_weights(MODEL_PATH)


@app.on_event("startup")
def startup_event():
    load_model()


@app.get("/")
def home():
    return {"status": "Real-ESRGAN API is running 🚀"}


@app.post("/enhance")
async def enhance_image(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        output = model.predict(image)

        buf = io.BytesIO()
        output.save(buf, format="PNG")
        buf.seek(0)

        return StreamingResponse(buf, media_type="image/png")

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
