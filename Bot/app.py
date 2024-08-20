import base64

import cv2
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os
import uvicorn
from passport_logic import get_info

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
async def read_index():
    return FileResponse('static/index.html')


@app.post("/upload-image/")
async def upload_image(file: UploadFile = File(...)):
    try:
        file_location = f"uploads/{file.filename}"
        with open(file_location, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        extracted_data, face_img, mrz_code_img = get_info(file_location)

        # Convert images to base64
        def encode_image_to_base64(image):
            _, buffer = cv2.imencode('.jpg', image)
            return base64.b64encode(buffer).decode('utf-8')

        face_img_base64 = encode_image_to_base64(face_img) if face_img is not None else None
        mrz_code_img_base64 = encode_image_to_base64(mrz_code_img)

        # Optionally, remove the temporary file after processing
        os.remove(file_location)

        return JSONResponse(content={
            'data': extracted_data,
            'face_image': face_img_base64,
            'mrz_image': mrz_code_img_base64
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=443,
        ssl_keyfile="ip_address.key",
        ssl_certfile="ip_address.crt"
    )

