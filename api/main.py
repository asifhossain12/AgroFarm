from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
import os
import logging


logging.basicConfig(level=logging.INFO, encoding='utf-8')
logger = logging.getLogger(__name__)

os.environ['PYTHONIOENCODING'] = 'utf-8'

app = FastAPI()

MODEL_PATH = "models/1.h5"

try:
    MODEL = tf.keras.models.load_model(MODEL_PATH) 
    logger.info("Model loaded successfully.")
    
    MODEL.summary()
except Exception as e:
    logger.error(f"Error loading the model: {e}")
    raise RuntimeError("Failed to load the model.") from e

CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return JSONResponse(content={"message": "Hello, I am alive"})

def read_file_as_image(data) -> np.ndarray:
    """Read and convert uploaded image file data into a NumPy array."""
    try:
        image = Image.open(BytesIO(data)).resize((256, 256))  
        return np.array(image)
    except Exception as e:
        logger.error(f"Error processing image: {e}")
        raise ValueError("Image processing error") from e

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        
        image = read_file_as_image(await file.read())
        img_batch = np.expand_dims(image, axis=0)  

        
        predictions = MODEL.predict(img_batch)
        predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
        confidence = np.max(predictions[0])

        
        response_content = {
            'class': predicted_class,
            'confidence': float(confidence)
        }

        return JSONResponse(content=response_content)

    except UnicodeEncodeError as e:
        logger.error(f"Unicode encoding error: {e}")
        return JSONResponse(content={
            'error': 'Encoding issue in prediction output',
            'details': str(e)
        })

    except ValueError as e:
        logger.error(f"Value error: {e}")
        return JSONResponse(content={
            'error': 'Image processing error',
            'details': str(e)
        })

    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return JSONResponse(content={
            'error': 'Unexpected error occurred during prediction',
            'details': str(e)
        })

if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)

# uvicorn main:app --host localhost --port 8000 --log-level info --workers 1
