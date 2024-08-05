from fastapi import FastAPI
from pydantic import BaseModel, HttpUrl
from fastapi.middleware.cors import CORSMiddleware
from services.genai import (YoutubeProcessor, GeminiProcessor)
import json
from . import servicekey


with open('servicekey.json', 'r') as f:
    data = json.load(f)

project_id = data['project_id']

class VideoAnalysisRequest(BaseModel):
    youtube_link : HttpUrl


app = FastAPI()

genai_processor = GeminiProcessor("gemini-1.0-pro-002",project_id) #Enter your project id here instead of abcde

#Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*']
)

@app.post("/analyze_video")
def analyze_video(request : VideoAnalysisRequest):

    processor = YoutubeProcessor(genai_processor=genai_processor)
    result = processor.retrieve_youtube_documents(str(request.youtube_link),verbose=True)

    # summary = genai_processor.generate_document_summary(result,verbose=True)
    
    #Find key concepts
    key_concepts = processor.find_key_concepts(result,verbose=True)

    return {
        "key_concepts" : key_concepts
    }