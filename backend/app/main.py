from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import tempfile
import os
import json
import requests
from dotenv import load_dotenv
from typing import Dict, Any
import imghdr

# Import the prediction function from your model file
from predict import predict_leaf_disease

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(title="Plant Disease Prediction API",
              description="API for predicting plant diseases from leaf images and providing treatment information")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Groq API Configuration
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "llama3-8b-8192"  # Replace with actual Groq model name

async def get_disease_info(disease_name: str) -> Dict[str, Any]:
    """Get disease information from Groq LLM"""
    if not GROQ_API_KEY:
        raise HTTPException(status_code=500, detail="Groq API key not configured")
    
    # Clean up disease name for better prompting
    clean_disease_name = disease_name.replace("___", " ").replace("_", " ")
    
    # Create headers for Groq API
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    
    # Create prompt for Groq
    prompt = f"""
    Provide the following information about the plant disease '{clean_disease_name}':
    1. Description: What is this disease and what causes it?
    2. Symptoms: What visual symptoms appear on the plant?
    3. Disease cycle: How does the disease progress?
    4. Treatment options: What are effective treatments?
    5. Prevention methods: How can farmers prevent this disease?
    
    Format the response as JSON with these sections as keys.
    """
    
    # Create payload for Groq API
    payload = {
        "model": GROQ_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3,
        "max_tokens": 800
    }
    
    try:
        # Make request to Groq API
        response = requests.post(GROQ_API_URL, headers=headers, json=payload)
        response.raise_for_status()
        
        # Parse response
        result = response.json()
        response_content = result["choices"][0]["message"]["content"]
        
        # Try to parse as JSON, but return text if not valid JSON
        try:
            return json.loads(response_content)
        except json.JSONDecodeError:
            return {"raw_content": response_content}
            
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Error calling Groq API: {str(e)}")

@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "Plant Disease Prediction API. Use /predict endpoint with an image file."}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Predict plant disease from image and provide detailed information
    
    - **file**: An image file of a plant leaf (supports jpg, png, and other common image formats)
    
    Returns prediction and disease information
    """
    try:
        # Read the file content
        content = await file.read()
        
        # Save to a temporary file for processing
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        try:
            # Use imghdr to check if it's a valid image
            img_type = imghdr.what(temp_file_path)
            if img_type is None:
                raise HTTPException(status_code=400, detail="Invalid image file")
            
            # Predict disease using the imported function
            prediction_result = predict_leaf_disease(temp_file_path)
            
            # Get disease information from LLM
            disease_info = await get_disease_info(prediction_result["category"])
            
            # Return combined results
            return JSONResponse(content={
                "filename": file.filename,
                "image_type": img_type,
                "prediction": {
                    "class": prediction_result["category"],
                    "confidence": prediction_result["confidence"]
                },
                "disease_information": disease_info
            })
        finally:
            # Clean up the temporary file
            os.unlink(temp_file_path)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)