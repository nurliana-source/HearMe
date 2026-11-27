# backend/api_server.py - VERSI FINAL (5 EMOSI + CHAT n8n + HAPUS RIWAYAT)

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import numpy as np
import librosa
import tensorflow as tf
import joblib
import os
import tempfile
import random
import requests
import json
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
YOUTUBE_API_KEY = os.getenv('YOUTUBE_API_KEY')
print(f"🔑 API Key loaded: {'Yes' if YOUTUBE_API_KEY else 'No'}")

# ============ CUSTOM LAYERS ============
class CustomAttention(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        kwargs.pop('use_scale', None)
        kwargs.pop('score_mode', None)
        kwargs.pop('dropout', None)
        super().__init__(**kwargs)
    
    def call(self, inputs):
        if isinstance(inputs, list):
            query = inputs[0]
            value = inputs[1] if len(inputs) > 1 else inputs[0]
        else:
            query = inputs
            value = inputs
        
        score = tf.matmul(query, value, transpose_b=True)
        attention = tf.nn.softmax(score)
        return tf.matmul(attention, value)
    
    def get_config(self):
        return super().get_config()


class CustomBidirectional(tf.keras.layers.Bidirectional):
    def __init__(self, layer, **kwargs):
        super().__init__(layer, **kwargs)
    
    def get_config(self):
        config = super().get_config()
        return config


tf.keras.utils.get_custom_objects().update({
    'Attention': CustomAttention,
    'Bidirectional': CustomBidirectional
})
# ====================================================

app = FastAPI(title="HearMe API", description="Speech Emotion Recognition API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Konfigurasi
SAMPLE_RATE = 22050
DURATION = 3
N_MFCC = 40
HOP_LENGTH = 512
N_FFT = 2048
TARGET_LENGTH = int(SAMPLE_RATE * DURATION / HOP_LENGTH)

# Ambang minimal energi suara (RMS) agar rekaman dianggap ada suara bicara.
# Bisa diubah tanpa mengedit kode: tambahkan MIN_RMS=0.01 di file .env (kalibrasi dari log "RMS rekaman").
MIN_RMS = float(os.getenv('MIN_RMS', '0.01'))

# Mode debug: simpan audio yang diterima backend ke folder debug_audio untuk didengarkan ulang saat pengujian.
# Nyalakan dengan DEBUG_SAVE_AUDIO=1 di file .env, dan MATIKAN (hapus atau 0) setelah selesai karena menyimpan suara pengguna.
DEBUG_SAVE_AUDIO = os.getenv('DEBUG_SAVE_AUDIO', '0') == '1'

# Path model
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEBUG_DIR = os.path.join(BASE_DIR, 'debug_audio')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
MODEL_PATH = os.path.join(MODEL_DIR, 'hearme_best_model.h5')
LABEL_ENCODER_PATH = os.path.join(MODEL_DIR, 'label_encoder.pkl')
NORM_PARAMS_PATH = os.path.join(MODEL_DIR, 'norm_params.pkl')
HISTORY_FILE = os.path.join(BASE_DIR, 'history_data.json')

# URL Webhook n8n (Bisa diatur di file .env)
N8N_WEBHOOK_URL = os.getenv('N8N_WEBHOOK_URL', 'http://localhost:5678/webhook/chat-voice')
N8N_TIMEOUT_SECONDS = 60

print(f"📁 Model path: {MODEL_PATH}")
print(f"🔗 n8n Webhook URL: {N8N_WEBHOOK_URL}")
print(f"🔈 Ambang RMS minimal: {MIN_RMS}")

# ============ ROUTER RIWAYAT ============
import threading
from datetime import timezone
from typing import Optional
from fastapi import APIRouter, Query

LOCAL_TZ = timezone(timedelta(hours=7))
VALID_EMOTIONS = ['happy', 'sad', 'angry', 'fear', 'neutral']
EMOTION_ALIAS = {'fearful': 'fear', 'happiness': 'happy', 'sadness': 'sad', 'anger': 'angry'}
MATCH_TOLERANCE_SECONDS = 5

def _norm_emotion(value):
    key = str(value or '').strip().lower()
    return EMOTION_ALIAS.get(key, key)

def _parse_ts(value):
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(str(value).strip().replace('Z', '+00:00'))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)

def _item_time(item):
    return _parse_ts(item.get('timestamp')) or _parse_ts(item.get('date'))

def create_history_router(history_file=HISTORY_FILE):
    router = APIRouter()
    lock = threading.Lock()

    def load():
        try:
            with open(history_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data if isinstance(data, list) else []
        except (FileNotFoundError, json.JSONDecodeError):
            return []

    def save(data):
        with open(history_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False)

    @router.get('/dashboard/stats')
    def dashboard_stats(days: int = 7, session_id: Optional[str] = None):
        days = max(1, days)
        now_utc = datetime.now(timezone.utc)
        cutoff = now_utc - timedelta(days=days)
        today_key = now_utc.astimezone(LOCAL_TZ).strftime('%Y-%m-%d')

        with lock:
            history = load()

        rows = []
        for item in history:
            if not isinstance(item, dict):
                continue
            if session_id and item.get('session_id') != session_id:
                continue
            emotion = _norm_emotion(item.get('emotion'))
            if emotion not in VALID_EMOTIONS:
                continue
            ts = _item_time(item)
            if ts is None or ts < cutoff:
                continue
            rows.append({
                'emotion': emotion,
                'confidence': item.get('confidence'),
                'session_id': item.get('session_id'),
                'ts_utc': ts,
                'ts_local': ts.astimezone(LOCAL_TZ),
            })

        rows.sort(key=lambda r: r['ts_utc'], reverse=True)

        emotion_count = {}
        daily_trends = {}
        hourly = [0] * 24
        today_count = 0

        for r in rows:
            emo = r['emotion']
            local = r['ts_local']
            day_key = local.strftime('%Y-%m-%d')

            emotion_count[emo] = emotion_count.get(emo, 0) + 1

            day = daily_trends.setdefault(day_key, {'total': 0, 'emotions': {}})
            day['total'] += 1
            day['emotions'][emo] = day['emotions'].get(emo, 0) + 1

            hourly[local.hour] += 1
            if day_key == today_key:
                today_count += 1

        confs = [float(r['confidence']) for r in rows if isinstance(r['confidence'], (int, float))]
        dominant = max(emotion_count, key=emotion_count.get) if emotion_count else 'neutral'

        return {
            'totalInteractions': len(rows),
            'avgAccuracy': (sum(confs) / len(confs)) if confs else 0,
            'dominantEmotion': dominant,
            'todayCount': today_count,
            'uniqueSessions': len({r['session_id'] for r in rows if r['session_id']}),
            'emotionCount': emotion_count,
            'dailyTrends': daily_trends,
            'hourlyDistribution': hourly,
            'recentActivity': [
                {
                    'emotion': r['emotion'],
                    'confidence': r['confidence'],
                    'timestamp': r['ts_utc'].strftime('%Y-%m-%dT%H:%M:%SZ'),
                }
                for r in rows[:10]
            ],
        }

    @router.delete('/history/clear')
    def history_clear(session_id: str = Query(..., min_length=1)):
        with lock:
            data = load()
            kept = [
                i for i in data
                if not (isinstance(i, dict) and i.get('session_id') == session_id)
            ]
            deleted = len(data) - len(kept)
            save(kept)
        return {'success': True, 'deleted': deleted}

    @router.delete('/history/item')
    def history_item_delete(
        session_id: str = Query(..., min_length=1),
        timestamp: str = Query(..., min_length=1),
        item_id: Optional[str] = Query(None, alias='id'),
    ):
        with lock:
            data = load()
            target_index = None

            if item_id:
                for idx, item in enumerate(data):
                    if (isinstance(item, dict)
                            and item.get('session_id') == session_id
                            and str(item.get('id')) == str(item_id)):
                        target_index = idx
                        break

            if target_index is None:
                target_time = _parse_ts(timestamp)
                best_diff = None
                if target_time is not None:
                    for idx, item in enumerate(data):
                        if not isinstance(item, dict) or item.get('session_id') != session_id:
                            continue
                        ts = _item_time(item)
                        if ts is None:
                            continue
                        diff = abs((ts - target_time).total_seconds())
                        if diff <= MATCH_TOLERANCE_SECONDS and (best_diff is None or diff < best_diff):
                            target_index, best_diff = idx, diff

            deleted = 0
            if target_index is not None:
                data.pop(target_index)
                save(data)
                deleted = 1

        return {'success': True, 'deleted': deleted}

    return router

app.include_router(create_history_router())
# ==============================================================================

# Load model
def load_model_with_fallback():
    try:
        model = tf.keras.models.load_model(
            MODEL_PATH,
            custom_objects={'Attention': CustomAttention, 'Bidirectional': CustomBidirectional},
            compile=False
        )
        print("✅ Model loaded successfully!")
        return model
    except Exception as e:
        print(f"⚠️ Error loading model: {e}")
        try:
            model = tf.keras.models.load_model(MODEL_PATH, compile=False)
            print("✅ Model loaded (fallback mode)!")
            return model
        except Exception as e2:
            print(f"❌ Fallback also failed: {e2}")
            return None

model = load_model_with_fallback()
if model is not None:
    print(f"🧠 Jumlah parameter model: {model.count_params()}")   # model baru: 955141

try:
    label_encoder = joblib.load(LABEL_ENCODER_PATH)
    print(f"✅ Label encoder loaded! Classes: {label_encoder.classes_}")
except Exception as e:
    print(f"❌ Error loading label encoder: {e}")
    label_encoder = None

try:
    norm_params = joblib.load(NORM_PARAMS_PATH)
    print(f"✅ Norm params loaded! mean={norm_params['mean']:.4f}, std={norm_params['std']:.4f}")
except Exception as e:
    print(f"⚠️ Error loading norm params: {e}")
    norm_params = None

# ============ HISTORY FUNCTIONS ============
def load_history():
    try:
        with open(HISTORY_FILE, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return []

def save_history_to_file(data):
    with open(HISTORY_FILE, 'w') as f:
        json.dump(data, f)

# ============ YOUTUBE API ============
def search_youtube_videos(query, max_results=3):
    print(f"🔍 SEARCHING YOUTUBE: {query}")
    if not YOUTUBE_API_KEY:
        return []
    url = "https://www.googleapis.com/youtube/v3/search"
    params = {
        "part": "snippet", "q": query, "maxResults": max_results,
        "type": "video", "key": YOUTUBE_API_KEY, "videoDuration": "medium", "safeSearch": "moderate"
    }
    try:
        response = requests.get(url, params=params, timeout=10)
        data = response.json()
        if response.status_code != 200:
            return []
        videos = []
        for item in data.get('items', []):
            video_id = item['id']['videoId']
            videos.append({
                'title': item['snippet']['title'],
                'url': f"https://www.youtube.com/watch?v={video_id}",
                'thumbnail': item['snippet']['thumbnails']['medium']['url']
            })
        return videos
    except Exception:
        return []

# ============ GET VIDEO QUERY ============
def get_video_query(emotion):
    queries = {
        'sad': ['funny videos compilation 2024', 'try not to laugh challenge', 'funny cats and dogs videos', 'best fails of the week'],
        'angry': ['relaxing music meditation', 'calming nature sounds', 'stress relief music', 'peaceful piano music'],
        'fear': ['motivational speech', 'confidence motivation', 'overcoming fear inspiration', 'daily motivation 2024'],
        'happy': ['happy songs upbeat', 'positive vibes music', 'feel good songs 2024', 'uplifting music playlist'],
        'neutral': ['daily motivation', 'inspirational videos', 'morning motivation', 'success motivation']
    }
    return random.choice(queries.get(emotion, queries['neutral']))

# ============ EMPATHETIC RESPONSES ============
EMPATHETIC_RESPONSES = {
    'happy': {
        'messages': ["Senang mendengar kamu bahagia! 🎉", "Keceriaanmu menular! Yuk rayakan!", "Kebahagiaan adalah hal terindah. Pertahankan ya! 😊"],
        'music': ["Happy - Pharrell Williams", "Can't Stop The Feeling - Justin Timberlake", "Walking on Sunshine - Katrina & The Waves", "Good Vibrations - The Beach Boys", "I'm Yours - Jason Mraz", "Beautiful Day - U2"],
        'activities': ["Bagikan kebahagiaan dengan teman", "Lakukan hobi favorit"]
    },
    'sad': {
        'messages': ["Aku di sini untukmu. Semua akan baik-baik saja 💙", "Menangis itu wajar. Ingin cerita?", "Hari ini mungkin berat, tapi kamu kuat! 🌈"],
        'music': ["Fix You - Coldplay", "Lean on Me - Bill Withers", "Someone Like You - Adele", "Let Her Go - Passenger", "Everybody Hurts - R.E.M.", "Skinny Love - Bon Iver"],
        'activities': ["Hubungi teman dekat", "Jalan-jalan sebentar", "Tulis jurnal"]
    },
    'angry': {
        'messages': ["Aku mengerti kamu marah. Ambil napas dalam-dalam 🧘", "Mari kita tenangkan diri sejenak", "Wajar merasa marah. Tapi ingat, kamu bisa mengendalikannya 💪"],
        'music': ["Let It Be - The Beatles", "Three Little Birds - Bob Marley", "Don't Worry Be Happy - Bobby McFerrin", "Here Comes The Sun - The Beatles", "What a Wonderful World - Louis Armstrong"],
        'activities': ["Hitung sampai 10", "Minum air putih", "Jalan santai"]
    },
    'fear': {
        'messages': ["Kamu aman di sini. Aku bersamamu 🤗", "Takut itu wajar. Tapi ingat, kamu lebih kuat dari yang kamu kira!", "Semua akan baik-baik saja. Percaya pada dirimu sendiri ✨"],
        'music': ["Don't Worry Be Happy - Bobby McFerrin", "Here Comes The Sun - The Beatles", "Three Little Birds - Bob Marley", "What a Wonderful World - Louis Armstrong", "Lean on Me - Bill Withers"],
        'activities': ["Tarik napas dalam 3 kali", "Dengar musik favorit"]
    },
    'neutral': {
        'messages': ["Terima kasih sudah berbagi perasaanmu! 👍", "Hari-hari biasa juga menyenangkan, ya!", "Tetap semangat menjalani hari! 💪"],
        'music': ["Good Day - Nappy Roots", "Best Day Of My Life - American Authors", "On Top of the World - Imagine Dragons", "Beautiful Day - U2", "Walking on Sunshine - Katrina & The Waves"],
        'activities': ["Lakukan sesuatu yang produktif", "Belajar hal baru"]
    }
}

def get_empathetic_response(emotion):
    data = EMPATHETIC_RESPONSES.get(emotion, EMPATHETIC_RESPONSES['neutral'])
    music_list = data.get('music', [])
    selected_music = random.sample(music_list, 3) if len(music_list) >= 3 else music_list
    return {
        'message': random.choice(data['messages']),
        'music_recommendations': selected_music,
        'activity_suggestions': data.get('activities', [])
    }

# ============ API ENDPOINTS ============

@app.get("/")
async def root():
    return {"message": "HearMe API is running!", "status": "healthy", "model_loaded": model is not None, "label_encoder_loaded": label_encoder is not None}

@app.post("/history/save")
async def save_history_endpoint(history_data: dict):
    try:
        history = load_history()
        history.append(history_data)
        save_history_to_file(history)
        return {"success": True, "message": "Data saved", "count": len(history)}
    except Exception as e:
        return {"success": False, "message": str(e)}

@app.get("/history/all")
async def get_all_history():
    try:
        return {"success": True, "data": load_history()}
    except Exception as e:
        return {"success": False, "data": [], "message": str(e)}

# ============ ENDPOINT CHAT KE n8n ============
def _extract_reply(data):
    if isinstance(data, list):
        data = data[0] if data else {}
    if isinstance(data, dict):
        for key in ('reply', 'output', 'text'):
            value = data.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return ''
    if isinstance(data, str):
        return data.strip()
    return ''


@app.post("/chat")
def chat_with_n8n(request: dict):
    message = str(request.get('message') or '').strip()
    emotion = request.get('emotion') or 'neutral'
    session_id = request.get('sessionId') or 'default'

    if not message:
        return JSONResponse(status_code=400, content={"error": "Pesan kosong"})

    payload = {"message": message, "emotion": emotion, "sessionId": session_id}
    print(f"📤 Meneruskan chat ke n8n: {message}")

    try:
        response = requests.post(
            N8N_WEBHOOK_URL,
            json=payload,
            timeout=N8N_TIMEOUT_SECONDS,
            headers={'ngrok-skip-browser-warning': 'true'},
        )
        print(f"📥 n8n status: {response.status_code}, isi: {response.text[:300]}")

        if response.status_code != 200:
            return JSONResponse(
                status_code=502,
                content={"error": f"AI Agent mengembalikan status {response.status_code}"},
            )

        try:
            data = response.json()
        except ValueError:
            data = response.text

        reply = _extract_reply(data)
        if not reply:
            return JSONResponse(status_code=502, content={"error": "AI Agent tidak mengembalikan jawaban"})

        return {"success": True, "reply": reply, "output": reply}

    except requests.exceptions.Timeout:
        print("❌ n8n terlalu lama menjawab")
        return JSONResponse(status_code=504, content={"error": "AI Agent terlalu lama menjawab"})
    except Exception as e:
        print(f"❌ Gagal menghubungi n8n: {e!r}")
        return JSONResponse(status_code=502, content={"error": str(e)})


@app.post("/predict")
async def predict(audio: UploadFile = File(...)):
    if model is None:
        return JSONResponse(status_code=500, content={"error": "Model tidak tersedia"})
    
    try:
        audio_bytes = await audio.read()
        if len(audio_bytes) < 1000:
            return JSONResponse(status_code=400, content={"error": "Audio terlalu pendek"})
        
        content_type = audio.content_type or ''
        file_ext = '.wav'
        if 'wav' in content_type: file_ext = '.wav'
        elif 'mp3' in content_type: file_ext = '.mp3'
        elif 'webm' in content_type: file_ext = '.webm'
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
            tmp_file.write(audio_bytes)
            tmp_path = tmp_file.name
        
        try:
            signal, sr = None, None
            
            try:
                from pydub import AudioSegment
                audio_segment = AudioSegment.from_file(tmp_path).set_frame_rate(SAMPLE_RATE).set_channels(1)[:DURATION * 1000]
                samples = np.array(audio_segment.get_array_of_samples())
                if audio_segment.sample_width == 2: samples = samples.astype(np.float32) / 32768.0
                elif audio_segment.sample_width == 4: samples = samples.astype(np.float32) / 2147483648.0
                signal, sr = samples, SAMPLE_RATE
            except:
                try:
                    signal, sr = librosa.load(tmp_path, sr=SAMPLE_RATE, duration=DURATION, mono=True)
                except:
                    import scipy.io.wavfile as wav
                    sr, signal = wav.read(tmp_path)
                    if signal.dtype == np.int16: signal = signal.astype(np.float32) / 32768.0
                    elif signal.dtype == np.int32: signal = signal.astype(np.float32) / 2147483648.0
            
            if signal is None or len(signal) == 0:
                return JSONResponse(status_code=400, content={"error": "Tidak dapat membaca audio"})

            # --- Pengecekan suara terlalu pelan (SEBELUM normalisasi amplitudo) ---
            rms = float(np.sqrt(np.mean(np.square(signal.astype(np.float64)))))
            print(f"🔈 RMS rekaman: {rms:.5f} (ambang {MIN_RMS})")
            if rms < MIN_RMS:
                return JSONResponse(
                    status_code=422,
                    content={"error": "Suara bicara tidak terdeteksi. Coba bicara lebih jelas dan dekat mikrofon.",
                             "rms": rms}
                )
            # ----------------------------------------------------------------------
            raw_signal = signal.copy()   # sinyal sebelum normalisasi (untuk mode debug)
            
            if np.max(np.abs(signal)) > 0: signal = signal / np.max(np.abs(signal))
            
            mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH).T
            if mfcc.shape[0] < TARGET_LENGTH:
                mfcc = np.pad(mfcc, ((0, TARGET_LENGTH - mfcc.shape[0]), (0, 0)), 'constant')
            else:
                mfcc = mfcc[:TARGET_LENGTH, :]
            
            if norm_params is not None:
                mfcc = (mfcc - norm_params['mean']) / (norm_params['std'] + 1e-8)
            
            predictions = model.predict(np.expand_dims(mfcc, axis=0), verbose=0)[0]
            predicted_idx = np.argmax(predictions)
            confidence = float(predictions[predicted_idx])
            emotion = label_encoder.inverse_transform([predicted_idx])[0] if label_encoder is not None else "unknown"
            probs = {str(c): round(float(p), 4) for c, p in zip(label_encoder.classes_, predictions)} if label_encoder is not None else {}
            print(f"📊 Probabilitas: {probs} -> {emotion}")

            if DEBUG_SAVE_AUDIO:
                try:
                    import soundfile as sf
                    os.makedirs(DEBUG_DIR, exist_ok=True)
                    base = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{emotion}_{confidence:.2f}_rms{rms:.3f}"
                    sf.write(os.path.join(DEBUG_DIR, base + '_didengar_model.wav'), raw_signal, sr)
                    with open(os.path.join(DEBUG_DIR, base + '_asli' + file_ext), 'wb') as f_dbg:
                        f_dbg.write(audio_bytes)
                    print(f"🐞 Audio debug disimpan: {base}")
                except Exception as e_dbg:
                    print(f"🐞 Gagal menyimpan audio debug: {e_dbg}")
            
            empathetic = get_empathetic_response(emotion)
            videos = search_youtube_videos(get_video_query(emotion))
            
            return {
                "success": True, "emotion": emotion, "confidence": confidence, "probabilities": probs,
                "message": empathetic['message'], "self_help_tips": empathetic['activity_suggestions'],
                "video_recommendations": videos, "music_recommendations": empathetic['music_recommendations']
            }
        finally:
            if os.path.exists(tmp_path):
                try:
                    import time, gc
                    time.sleep(0.3); gc.collect(); os.unlink(tmp_path)
                except: pass
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)