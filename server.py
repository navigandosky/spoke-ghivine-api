from fastapi import FastAPI, APIRouter, UploadFile, File, HTTPException, Form
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
from pathlib import Path
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional, Dict
import uuid
from datetime import datetime, timezone
import aiofiles

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

mongo_url = os.environ.get('MONGO_URL', '')
db_name = os.environ.get('DB_NAME', 'spoke_ghivine')
client = None
db = None

if mongo_url:
    client = AsyncIOMotorClient(mongo_url)
    db = client[db_name]

UPLOADS_DIR = ROOT_DIR / 'uploads'
UPLOADS_DIR.mkdir(exist_ok=True)
(UPLOADS_DIR / 'images').mkdir(exist_ok=True)
(UPLOADS_DIR / 'audio').mkdir(exist_ok=True)

app = FastAPI(title="Spoke Ghivine API")
api_router = APIRouter(prefix="/api")

class TranslatedText(BaseModel):
    it: str = ""
    en: str = ""
    fr: str = ""
    de: str = ""

class Space(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: TranslatedText
    description: TranslatedText
    matterport_model_id: str = ""
    external_tour_url: str = ""
    images: List[str] = []
    is_active: bool = True
    order: int = 0
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

class SpaceCreate(BaseModel):
    name: TranslatedText
    description: TranslatedText
    matterport_model_id: str = ""
    external_tour_url: str = ""
    images: List[str] = []
    is_active: bool = True
    order: int = 0

class SpaceUpdate(BaseModel):
    name: Optional[TranslatedText] = None
    description: Optional[TranslatedText] = None
    matterport_model_id: Optional[str] = None
    external_tour_url: Optional[str] = None
    images: Optional[List[str]] = None
    is_active: Optional[bool] = None
    order: Optional[int] = None

class POI(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    space_id: str
    poi_number: int
    name: TranslatedText
    description: TranslatedText
    audio_files: Dict[str, str] = {}
    mattertag_id: Optional[str] = None
    position: Optional[Dict] = None
    is_active: bool = True
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

class POICreate(BaseModel):
    space_id: str
    poi_number: int
    name: TranslatedText
    description: TranslatedText
    mattertag_id: Optional[str] = None
    position: Optional[Dict] = None
    is_active: bool = True

class POIUpdate(BaseModel):
    name: Optional[TranslatedText] = None
    description: Optional[TranslatedText] = None
    mattertag_id: Optional[str] = None
    position: Optional[Dict] = None
    is_active: Optional[bool] = None
    audio_files: Optional[Dict[str, str]] = None

class TranslationRequest(BaseModel):
    text: str
    source_lang: str = "it"
    target_lang: str

class TTSRequest(BaseModel):
    text: str
    language: str
    poi_id: str

@api_router.get("/")
async def root():
    return {"message": "Spoke Ghivine API", "version": "1.0.0"}

@api_router.get("/spaces", response_model=List[Space])
async def get_spaces():
    if db is None:
        return []
    spaces = await db.spaces.find({}, {"_id": 0}).sort("order", 1).to_list(100)
    return spaces

@api_router.get("/spaces/{space_id}", response_model=Space)
async def get_space(space_id: str):
    if db is None:
        raise HTTPException(status_code=500, detail="Database not configured")
    space = await db.spaces.find_one({"id": space_id}, {"_id": 0})
    if not space:
        raise HTTPException(status_code=404, detail="Space not found")
    return space

@api_router.post("/spaces", response_model=Space)
async def create_space(space: SpaceCreate):
    if db is None:
        raise HTTPException(status_code=500, detail="Database not configured")
    space_obj = Space(**space.model_dump())
    doc = space_obj.model_dump()
    await db.spaces.insert_one(doc)
    return space_obj

@api_router.put("/spaces/{space_id}", response_model=Space)
async def update_space(space_id: str, space: SpaceUpdate):
    if db is None:
        raise HTTPException(status_code=500, detail="Database not configured")
    update_data = {k: v for k, v in space.model_dump().items() if v is not None}
    update_data['updated_at'] = datetime.now(timezone.utc).isoformat()
    result = await db.spaces.find_one_and_update(
        {"id": space_id},
        {"$set": update_data},
        return_document=True
    )
    if not result:
        raise HTTPException(status_code=404, detail="Space not found")
    result.pop('_id', None)
    return result

@api_router.delete("/spaces/{space_id}")
async def delete_space(space_id: str):
    if db is None:
        raise HTTPException(status_code=500, detail="Database not configured")
    result = await db.spaces.delete_one({"id": space_id})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Space not found")
    await db.pois.delete_many({"space_id": space_id})
    return {"message": "Space deleted successfully"}

@api_router.get("/pois", response_model=List[POI])
async def get_pois(space_id: Optional[str] = None):
    if db is None:
        return []
    query = {}
    if space_id:
        query["space_id"] = space_id
    pois = await db.pois.find(query, {"_id": 0}).sort("poi_number", 1).to_list(500)
    return pois

@api_router.get("/pois/{poi_id}", response_model=POI)
async def get_poi(poi_id: str):
    if db is None:
        raise HTTPException(status_code=500, detail="Database not configured")
    poi = await db.pois.find_one({"id": poi_id}, {"_id": 0})
    if not poi:
        raise HTTPException(status_code=404, detail="POI not found")
    return poi

@api_router.post("/pois", response_model=POI)
async def create_poi(poi: POICreate):
    if db is None:
        raise HTTPException(status_code=500, detail="Database not configured")
    poi_obj = POI(**poi.model_dump())
    doc = poi_obj.model_dump()
    await db.pois.insert_one(doc)
    return poi_obj

@api_router.put("/pois/{poi_id}", response_model=POI)
async def update_poi(poi_id: str, poi: POIUpdate):
    if db is None:
        raise HTTPException(status_code=500, detail="Database not configured")
    update_data = {k: v for k, v in poi.model_dump().items() if v is not None}
    update_data['updated_at'] = datetime.now(timezone.utc).isoformat()
    result = await db.pois.find_one_and_update(
        {"id": poi_id},
        {"$set": update_data},
        return_document=True
    )
    if not result:
        raise HTTPException(status_code=404, detail="POI not found")
    result.pop('_id', None)
    return result

@api_router.delete("/pois/{poi_id}")
async def delete_poi(poi_id: str):
    if db is None:
        raise HTTPException(status_code=500, detail="Database not configured")
    result = await db.pois.delete_one({"id": poi_id})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="POI not found")
    return {"message": "POI deleted successfully"}

@api_router.post("/translate")
async def translate_text(request: TranslationRequest):
    try:
        from emergentintegrations.llm.chat import LlmChat, UserMessage
        api_key = os.environ.get('EMERGENT_LLM_KEY')
        if not api_key:
            raise HTTPException(status_code=500, detail="Translation API key not configured")
        lang_names = {"it": "Italian", "en": "English", "fr": "French", "de": "German"}
        chat = LlmChat(
            api_key=api_key,
            session_id=f"translation-{uuid.uuid4()}",
            system_message=f"You are a professional translator. Translate the following text from {lang_names.get(request.source_lang, 'Italian')} to {lang_names.get(request.target_lang, 'English')}. Only provide the translation, no explanations."
        ).with_model("openai", "gpt-4o-mini")
        user_message = UserMessage(text=request.text)
        translation = await chat.send_message(user_message)
        return {"translation": translation, "source_lang": request.source_lang, "target_lang": request.target_lang}
    except Exception as e:
        logging.error(f"Translation error: {e}")
        raise HTTPException(status_code=500, detail=f"Translation failed: {str(e)}")

@api_router.post("/tts")
async def generate_audio(request: TTSRequest):
    try:
        from emergentintegrations.llm.openai import OpenAITextToSpeech
        api_key = os.environ.get('EMERGENT_LLM_KEY')
        if not api_key:
            raise HTTPException(status_code=500, detail="TTS API key not configured")
        voice_map = {"it": "nova", "en": "alloy", "fr": "shimmer", "de": "echo"}
        tts = OpenAITextToSpeech(api_key=api_key)
        audio_bytes = await tts.generate_speech(
            text=request.text,
            model="tts-1",
            voice=voice_map.get(request.language, "alloy"),
            response_format="mp3"
        )
        filename = f"{request.poi_id}_{request.language}_{uuid.uuid4().hex[:8]}.mp3"
        filepath = UPLOADS_DIR / 'audio' / filename
        async with aiofiles.open(filepath, 'wb') as f:
            await f.write(audio_bytes)
        audio_url = f"/api/uploads/audio/{filename}"
        if db:
            await db.pois.update_one(
                {"id": request.poi_id},
                {"$set": {f"audio_files.{request.language}": audio_url, "updated_at": datetime.now(timezone.utc).isoformat()}}
            )
        return {"audio_url": audio_url, "language": request.language}
    except Exception as e:
        logging.error(f"TTS error: {e}")
        raise HTTPException(status_code=500, detail=f"Audio generation failed: {str(e)}")

@api_router.post("/upload/image")
async def upload_image(file: UploadFile = File(...)):
    try:
        ext = Path(file.filename).suffix.lower()
        if ext not in ['.jpg', '.jpeg', '.png', '.webp', '.gif']:
            raise HTTPException(status_code=400, detail="Invalid image format")
        filename = f"{uuid.uuid4().hex}{ext}"
        filepath = UPLOADS_DIR / 'images' / filename
        async with aiofiles.open(filepath, 'wb') as f:
            content = await file.read()
            await f.write(content)
        return {"url": f"/api/uploads/images/{filename}"}
    except Exception as e:
        logging.error(f"Image upload error: {e}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@api_router.post("/upload/audio")
async def upload_audio(file: UploadFile = File(...), poi_id: str = Form(...), language: str = Form(...)):
    try:
        ext = Path(file.filename).suffix.lower()
        if ext not in ['.mp3', '.wav', '.ogg', '.m4a']:
            raise HTTPException(status_code=400, detail="Invalid audio format")
        filename = f"{poi_id}_{language}_{uuid.uuid4().hex[:8]}{ext}"
        filepath = UPLOADS_DIR / 'audio' / filename
        async with aiofiles.open(filepath, 'wb') as f:
            content = await file.read()
            await f.write(content)
        audio_url = f"/api/uploads/audio/{filename}"
        if db:
            await db.pois.update_one(
                {"id": poi_id},
                {"$set": {f"audio_files.{language}": audio_url, "updated_at": datetime.now(timezone.utc).isoformat()}}
            )
        return {"url": audio_url, "language": language}
    except Exception as e:
        logging.error(f"Audio upload error: {e}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@api_router.get("/config")
async def get_config():
    return {
        "matterport_sdk_key": os.environ.get('MATTERPORT_SDK_KEY', ''),
        "languages": ["it", "en", "fr", "de"],
        "language_names": {"it": "Italiano", "en": "English", "fr": "Français", "de": "Deutsch"}
    }

app.include_router(api_router)
app.mount("/api/uploads", StaticFiles(directory=str(UPLOADS_DIR)), name="uploads")

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=os.environ.get('CORS_ORIGINS', '*').split(','),
    allow_methods=["*"],
    allow_headers=["*"],
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

@app.on_event("shutdown")
async def shutdown_db_client():
    if client:
        client.close()
