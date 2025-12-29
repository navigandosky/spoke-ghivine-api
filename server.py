Ecco i 4 file da creare. Copia tutto:

---

**FILE 1: `server.py`**
Action: $ cat /app/backend/server.py
Observation: from fastapi import FastAPI, APIRouter, UploadFile, File, HTTPException, Form
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
import base64

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB connection
mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

# Create uploads directory
UPLOADS_DIR = ROOT_DIR / 'uploads'
UPLOADS_DIR.mkdir(exist_ok=True)
(UPLOADS_DIR / 'images').mkdir(exist_ok=True)
(UPLOADS_DIR / 'audio').mkdir(exist_ok=True)

# Create the main app
app = FastAPI(title="Spoke Ghivine API")

# Create a router with the /api prefix
api_router = APIRouter(prefix="/api")

# --- Models ---

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
    matterport_model_id: str
    external_tour_url: str = ""  # Link esterno (mpskin, overlay, ecc.)
    images: List[str] = []
    is_active: bool = True
    order: int = 0
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

class SpaceCreate(BaseModel):
    name: TranslatedText
    description: TranslatedText
    matterport_model_id: str
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
    audio_files: Dict[str, str] = {}  # {lang: audio_url}
    mattertag_id: Optional[str] = None
    position: Optional[Dict] = None  # x, y, z coordinates
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

# --- API Endpoints ---

@api_router.get("/")
async def root():
    return {"message": "Spoke Ghivine API", "version": "1.0.0"}

# Spaces endpoints
@api_router.get("/spaces", response_model=List[Space])
async def get_spaces():
    spaces = await db.spaces.find({}, {"_id": 0}).sort("order", 1).to_list(100)
    return spaces

@api_router.get("/spaces/{space_id}", response_model=Space)
async def get_space(space_id: str):
    space = await db.spaces.find_one({"id": space_id}, {"_id": 0})
    if not space:
        raise HTTPException(status_code=404, detail="Space not found")
    return space

@api_router.post("/spaces", response_model=Space)
async def create_space(space: SpaceCreate):
    space_obj = Space(**space.model_dump())
    doc = space_obj.model_dump()
    await db.spaces.insert_one(doc)
    return space_obj

@api_router.put("/spaces/{space_id}", response_model=Space)
async def update_space(space_id: str, space: SpaceUpdate):
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
    result = await db.spaces.delete_one({"id": space_id})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Space not found")
    # Also delete related POIs
    await db.pois.delete_many({"space_id": space_id})
    return {"message": "Space deleted successfully"}

# POI endpoints
@api_router.get("/pois", response_model=List[POI])
async def get_pois(space_id: Optional[str] = None):
    query = {}
    if space_id:
        query["space_id"] = space_id
    pois = await db.pois.find(query, {"_id": 0}).sort("poi_number", 1).to_list(500)
    return pois

@api_router.get("/pois/{poi_id}", response_model=POI)
async def get_poi(poi_id: str):
    poi = await db.pois.find_one({"id": poi_id}, {"_id": 0})
    if not poi:
        raise HTTPException(status_code=404, detail="POI not found")
    return poi

@api_router.post("/pois", response_model=POI)
async def create_poi(poi: POICreate):
    poi_obj = POI(**poi.model_dump())
    doc = poi_obj.model_dump()
    await db.pois.insert_one(doc)
    return poi_obj

@api_router.put("/pois/{poi_id}", response_model=POI)
async def update_poi(poi_id: str, poi: POIUpdate):
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
    result = await db.pois.delete_one({"id": poi_id})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="POI not found")
    return {"message": "POI deleted successfully"}

# Translation endpoint
@api_router.post("/translate")
async def translate_text(request: TranslationRequest):
    try:
        from emergentintegrations.llm.chat import LlmChat, UserMessage
        
        api_key = os.environ.get('EMERGENT_LLM_KEY')
        if not api_key:
            raise HTTPException(status_code=500, detail="Translation API key not configured")
        
        lang_names = {
            "it": "Italian",
            "en": "English", 
            "fr": "French",
            "de": "German"
        }
        
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

# TTS endpoint
@api_router.post("/tts")
async def generate_audio(request: TTSRequest):
    try:
        from emergentintegrations.llm.openai import OpenAITextToSpeech
        
        api_key = os.environ.get('EMERGENT_LLM_KEY')
        if not api_key:
            raise HTTPException(status_code=500, detail="TTS API key not configured")
        
        # Voice mapping by language
        voice_map = {
            "it": "nova",
            "en": "alloy",
            "fr": "shimmer",
            "de": "echo"
        }
        
        tts = OpenAITextToSpeech(api_key=api_key)
        audio_bytes = await tts.generate_speech(
            text=request.text,
            model="tts-1",
            voice=voice_map.get(request.language, "alloy"),
            response_format="mp3"
        )
        
        # Save audio file
        filename = f"{request.poi_id}_{request.language}_{uuid.uuid4().hex[:8]}.mp3"
        filepath = UPLOADS_DIR / 'audio' / filename
        
        async with aiofiles.open(filepath, 'wb') as f:
            await f.write(audio_bytes)
        
        audio_url = f"/api/uploads/audio/{filename}"
        
        # Update POI with audio file
        await db.pois.update_one(
            {"id": request.poi_id},
            {"$set": {f"audio_files.{request.language}": audio_url, "updated_at": datetime.now(timezone.utc).isoformat()}}
        )
        
        return {"audio_url": audio_url, "language": request.language}
    except Exception as e:
        logging.error(f"TTS error: {e}")
        raise HTTPException(status_code=500, detail=f"Audio generation failed: {str(e)}")

# File upload endpoints
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
        
        # Update POI with audio file
        await db.pois.update_one(
            {"id": poi_id},
            {"$set": {f"audio_files.{language}": audio_url, "updated_at": datetime.now(timezone.utc).isoformat()}}
        )
        
        return {"url": audio_url, "language": language}
    except Exception as e:
        logging.error(f"Audio upload error: {e}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

# Config endpoint (public)
@api_router.get("/config")
async def get_config():
    return {
        "matterport_sdk_key": os.environ.get('MATTERPORT_SDK_KEY', ''),
        "languages": ["it", "en", "fr", "de"],
        "language_names": {
            "it": "Italiano",
            "en": "English",
            "fr": "Français",
            "de": "Deutsch"
        }
    }

# Seed initial data
@api_router.post("/seed")
async def seed_data():
    # Check if data already exists
    count = await db.spaces.count_documents({})
    if count > 0:
        return {"message": "Data already seeded"}
    
    # Create initial spaces
    spaces_data = [
        {
            "id": str(uuid.uuid4()),
            "name": {
                "it": "Grotta del Bue Marino",
                "en": "Bue Marino Cave",
                "fr": "Grotte du Bue Marino",
                "de": "Grotta del Bue Marino Höhle"
            },
            "description": {
                "it": "La Grotta del Bue Marino è una delle meraviglie naturali più spettacolari della Sardegna. Situata nel Golfo di Orosei, questa grotta carsica si estende per diversi chilometri lungo la costa, offrendo uno straordinario spettacolo di stalattiti e stalagmiti.",
                "en": "The Bue Marino Cave is one of the most spectacular natural wonders of Sardinia. Located in the Gulf of Orosei, this karst cave extends for several kilometers along the coast, offering an extraordinary display of stalactites and stalagmites.",
                "fr": "La Grotte du Bue Marino est l'une des merveilles naturelles les plus spectaculaires de la Sardaigne. Située dans le golfe d'Orosei, cette grotte karstique s'étend sur plusieurs kilomètres le long de la côte.",
                "de": "Die Grotta del Bue Marino ist eines der spektakulärsten Naturwunder Sardiniens. Im Golf von Orosei gelegen, erstreckt sich diese Karsthöhle über mehrere Kilometer entlang der Küste."
            },
            "matterport_model_id": "UqskS4cg92b",
            "images": [
                "https://www.grottabuemarino.com/wp-content/uploads/2024/11/f86cc4a8-294b-4675-916f-507ed7e71b45.webp",
                "https://www.grottabuemarino.com/wp-content/uploads/2024/11/76848ba9-7477-4f8f-823a-a0d30cc9f5da.webp"
            ],
            "is_active": True,
            "order": 1,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat()
        },
        {
            "id": str(uuid.uuid4()),
            "name": {
                "it": "Grotta Ispinigoli",
                "en": "Ispinigoli Cave",
                "fr": "Grotte d'Ispinigoli",
                "de": "Ispinigoli Höhle"
            },
            "description": {
                "it": "La Grotta di Ispinigoli è famosa per ospitare una delle stalagmiti più alte d'Europa, con i suoi 38 metri di altezza. Questa meraviglia geologica si erge maestosa nella sala principale.",
                "en": "The Ispinigoli Cave is famous for housing one of the tallest stalagmites in Europe, at 38 meters high. This geological wonder stands majestically in the main hall.",
                "fr": "La Grotte d'Ispinigoli est célèbre pour abriter l'une des plus hautes stalagmites d'Europe, avec ses 38 mètres de haut.",
                "de": "Die Ispinigoli-Höhle ist berühmt für eine der höchsten Stalagmiten Europas mit 38 Metern Höhe."
            },
            "matterport_model_id": "",
            "images": [],
            "is_active": True,
            "order": 2,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat()
        },
        {
            "id": str(uuid.uuid4()),
            "name": {
                "it": "Museo Archeologico di Dorgali",
                "en": "Archaeological Museum of Dorgali",
                "fr": "Musée Archéologique de Dorgali",
                "de": "Archäologisches Museum Dorgali"
            },
            "description": {
                "it": "Il Museo Archeologico di Dorgali custodisce un prezioso patrimonio che testimonia millenni di storia della Sardegna, dal Neolitico all'epoca medievale.",
                "en": "The Archaeological Museum of Dorgali holds a precious heritage that testifies to millennia of Sardinian history, from the Neolithic to the medieval period.",
                "fr": "Le Musée Archéologique de Dorgali conserve un précieux patrimoine témoignant de millénaires d'histoire de la Sardaigne.",
                "de": "Das Archäologische Museum von Dorgali bewahrt ein wertvolles Erbe, das Jahrtausende sardischer Geschichte bezeugt."
            },
            "matterport_model_id": "",
            "images": [],
            "is_active": True,
            "order": 3,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat()
        }
    ]
    
    await db.spaces.insert_many(spaces_data)
    return {"message": "Data seeded successfully", "spaces_count": len(spaces_data)}

# Include the router in the main app
app.include_router(api_router)

# Mount static files for uploads
app.mount("/api/uploads", StaticFiles(directory=str(UPLOADS_DIR)), name="uploads")

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=os.environ.get('CORS_ORIGINS', '*').split(','),
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()
Exit code: 0
