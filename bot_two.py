import asyncio
import json
import logging
import os
import re
import uuid
from asyncio import Semaphore
from collections import deque
from datetime import datetime
from functools import lru_cache
from typing import Deque, Dict, List, Optional
import httpx
import torch
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, util

logging.basicConfig(level=logging.INFO)

DATASET_PATH = "dataset.txt"
LOG_FILE = "chatbot_log.json"

app = FastAPI()
model_loaded = False
client = httpx.AsyncClient()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
semantic_cache = []
SIMILARITY_THRESHOLD = 0.85  # Lowered to capture more paraphrases
client = httpx.AsyncClient(timeout=80.0)
semaphore = Semaphore(10)

# دیکشنری برای ذخیره اتصال‌های فعال
active_connections: Dict[str, WebSocket] = {}

# کلاس برای ذخیره تاریخچه پاسخ‌ها برای هر کاربر
class UserHistory:
    def __init__(self, max_history=5):
        self.max_history = max_history
        self.history: Deque[Dict] = deque(maxlen=max_history)
    
    def add_interaction(self, question: str, answer: str):
        self.history.append({
            "question": question,
            "answer": answer,
            "timestamp": datetime.now().isoformat()
        })
    
    def get_history(self, count: int = None) -> List[Dict]:
        """تاریخچه پاسخ‌ها را برمی‌گرداند"""
        count = count or self.max_history
        return list(self.history)[-min(count, len(self.history)):]

# دیکشنری برای ذخیره تاریخچه پاسخ‌های کاربران
user_histories: Dict[str, UserHistory] = {}

# تابع برای دریافت یا ایجاد تاریخچه کاربر
def get_user_history(client_id: str) -> UserHistory:
    if client_id not in user_histories:
        user_histories[client_id] = UserHistory()
    return user_histories[client_id]

# تابع برای ذخیره لاگ در فایل JSON
def log_interaction(client_id: str, user_input: str, response: str):
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "client_id": client_id or "unknown",
        "question": user_input,
        "answer": response
    }
    
    # ذخیره در تاریخچه کاربر
    if client_id:
        history = get_user_history(client_id)
        history.add_interaction(user_input, response)
    
    try:
        logs = []
        if os.path.exists(LOG_FILE):
            try:
                with open(LOG_FILE, "r", encoding="utf-8") as f:
                    content = f.read().strip()
                    if content:
                        logs = json.loads(content)
            except json.JSONDecodeError as e:
                logging.warning(f"فایل لاگ نامعتبر است، فایل بازنویسی می‌شود: {str(e)}")
                logs = []
        
        logs.append(log_entry)
        with open(LOG_FILE, "w", encoding="utf-8") as f:
            json.dump(logs, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logging.error(f"خطا در ذخیره لاگ: {str(e)}")

def normalize_text(text):
    slang_dict = {
        "میصرفه": "مقرون به صرفه است",
        "می‌صرفه": "مقرون به صرفه است",
        "چیه": "چیست",
        "چطوری": "چگونه",
        "روشش چیه": "نحوه آن چیست",
        "فایده داره": "مزایا دارد",
        "بدیش چیه": "معایب آن چیست"
    }
    for slang, formal in slang_dict.items():
        text = text.replace(slang, formal)
    return text

def preprocess_question(user_input):
    """نرمال‌سازی متن با حفظ کلمات کلیدی برای بهبود تطبیق معنایی"""
    user_input = user_input.lower().strip()
    user_input = normalize_text(user_input)
    user_input = re.sub(r'[^\w\s؟!]', '', user_input)
    return user_input

def load_dataset(file_path):
    qa_pairs = []
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
        qa_blocks = re.split(r'\n{2,}', content)
        for block in qa_blocks:
            lines = block.strip().split("\n")
            if len(lines) >= 2:
                question = lines[0].strip()
                answer = " ".join(lines[1:]).strip()
                qa_pairs.append({"question": question, "answer": answer})
    return qa_pairs

@app.on_event("startup")
async def startup_event():
    global dataset, questions, question_embeddings
    dataset = load_dataset(DATASET_PATH)
    questions = [item["question"] for item in dataset]
    question_embeddings = model.encode(questions, convert_to_tensor=True)
    # Initialize chatbot_log.json
    if not os.path.exists(LOG_FILE) or os.path.getsize(LOG_FILE) == 0:
        with open(LOG_FILE, "w", encoding="utf-8") as f:
            json.dump([], f, ensure_ascii=False, indent=2)

@lru_cache(maxsize=1000)
def get_best_match(user_input):
    user_embedding = model.encode(user_input, convert_to_tensor=True)
    similarities = util.pytorch_cos_sim(user_embedding, question_embeddings)[0]
    max_idx = similarities.argmax().item()
    max_sim = similarities[max_idx].item()
    return dataset[max_idx], max_sim, user_embedding

def search_in_semantic_cache(user_embedding):
    """جستجوی کش معنایی با ثبت لاگ برای تطبیق‌ها و عدم تطبیق‌ها"""
    best_match = None
    best_similarity = 0.0
    for item in semantic_cache:
        # نرمال‌سازی بردارها برای مقایسه دقیق‌تر
        similarity = util.pytorch_cos_sim(user_embedding / user_embedding.norm(), 
                                       item["embedding"] / item["embedding"].norm())[0].item()
        if similarity > best_similarity:
            best_similarity = similarity
            best_match = item
    if best_match and best_similarity >= SIMILARITY_THRESHOLD:
        logging.info(f"تطبیق کش یافت شد: سؤال='{best_match['question']}', شباهت={best_similarity:.3f}")
        return best_match["answer"], best_match["question"]
    else:
        logging.debug(f"تطبیق کش یافت نشد: بهترین شباهت={best_similarity:.3f}")
        return None, None

def handle_default_responses(user_input: str) -> str | None:
    normalized = user_input.strip().lower()
    if normalized in ["سلام", "سلام!"]:
        return "سلام! چه کمکی می‌تونم بکنم؟"
    if normalized in ["خوبی؟", "حالت خوبه؟", "خوبی"]:
        return "من خوبم، ممنون که پرسیدی."
    if normalized in ["تو کی هستی؟", "تو کی هستی", "اسمت چیه؟", "کی هستی؟"]:
        return "من دستیار هوشمند هومنگرم که بهت کمک می‌کنم سرمایت رو بهتر مدیریت کنی و حفظ کنی."
    
    return None

def add_to_semantic_cache(question, answer, embedding):
    """اضافه کردن به کش معنایی با نرمال‌سازی بردار"""
    # نرمال‌سازی بردار جاسازی
    normalized_embedding = embedding / embedding.norm()
    semantic_cache.append({
        "question": question,
        "answer": answer,
        "embedding": normalized_embedding,
        "timestamp": datetime.now().isoformat()
    })
    
    # محدود کردن اندازه کش
    if len(semantic_cache) > 1000:
        semantic_cache.pop(0)
        logging.debug("قدیمی‌ترین ورودی کش حذف شد")

def generate_prompt_with_history(question: str, best_pair: dict, client_id: str = None, history_count: int = 2):
    prompt_base = (
        f"متن:\n{best_pair['answer']}\n\n"
        f"سؤال: {question}\n\n"
    )
    
    history_text = ""
    if client_id and client_id in user_histories:
        history = user_histories[client_id].get_history(history_count)
        if history:
            history_text = "تاریخچه گفتگوهای اخیر:\n"
            for i, item in enumerate(history):
                history_text += f"سؤال {i+1}: {item['question']}\n"
                history_text += f"پاسخ {i+1}: {item['answer']}\n\n"
    
    prompt_instructions = (
        f"دستورات:\n"
        f"1. پاسخ را فقط به زبان فارسی و بدون هیچ کلمه انگلیسی بنویس.\n"
        f"2. پاسخ باید کوتاه (حداکثر دو جمله)، روان، مفهومی و کاملاً بر اساس متن ارائه‌شده باشد.\n"
        f"3. از تاریخچه گفتگوهای قبلی برای ایجاد پاسخ بهتر و منسجم‌تر استفاده کن.\n"
        f"4. اگر سؤال کاربر مرتبط با گفتگوهای قبلی است، پاسخ‌های قبلی را در نظر بگیر.\n"
        f"5. از عباراتی مثل end_of_text یا تگ‌های برنامه‌نویسی استفاده نکن.\n"
        f"6. هیچ غلط املایی نداشته باش و تمام کلمات را صحیح بنویس.\n"
        f"7. اگر سؤال نامربوط یا خارج از متن بود، فقط بنویس: «اطلاعی ندارم.»\n"
        f"8. اگر سؤال درباره کلاهبرداری بود، فقط بنویس: «خیر، شرکت هومنگر با مجوز رسمی و پشتیبانی حقوقی، تضمین می‌دهد که این روش کاملاً قانونی و معتبر است.»\n"
        f"9. اگر سؤال درباره سودآوری یا فایده بود، فقط بنویس: «بله، فروش متری املاک یکی از روش‌های نوین سرمایه‌گذاری با سود بالا و ریسک پایین است.»\n"
        f"10. نقش شرکت هومنگر را مثبت و مطمئن نشان بده و تأکید کن که هومنگر بستری امن و شفاف برای خرید متری فراهم کرده است.\n"
        f"11. اگر پاسخ به هومنگر مرتبط بود، در انتها اضافه کن: «برای شروع فقط کافیه در سایت هومنگر ثبت‌نام کنی.»\n"
    )
    
    return prompt_base + (history_text if history_text else "") + prompt_instructions

class QuestionRequest(BaseModel):
    question: str
    client_id: Optional[str] = None
    history_count: Optional[int] = 2

@app.post("/ask")
async def ask_endpoint(request: QuestionRequest):
    try:
        async with semaphore:
            user_input = request.question.strip()
            client_id = request.client_id
            history_count = request.history_count if request.history_count is not None else 2
            
            if not user_input:
                raise HTTPException(status_code=400, detail="لطفاً سوالی وارد کنید.")

            default_response = handle_default_responses(user_input)
            if default_response:
                log_interaction(client_id, user_input, default_response)
                return {
                    "original_input": user_input,
                    "answers": [{
                        "input": user_input,
                        "matched_question": "پاسخ پیش‌فرض",
                        "answer": default_response,
                        "similarity": 1.0
                    }]
                }

            # استفاده از سؤال اصلی برای جاسازی معنایی
            user_embedding = model.encode(user_input, convert_to_tensor=True)
            cached_answer, cached_question = search_in_semantic_cache(user_embedding)
            if cached_answer:
                log_interaction(client_id, user_input, cached_answer)
                return {
                    "original_input": user_input,
                    "answers": [{
                        "input": user_input,
                        "matched_question": cached_question or "از حافظه کش پاسخ داده شد",
                        "answer": cached_answer,
                        "similarity": SIMILARITY_THRESHOLD
                    }]
                }

            preprocessed = preprocess_question(user_input)
            best_pair, similarity, _ = get_best_match(preprocessed)

            prompt = generate_prompt_with_history(user_input, best_pair, client_id, history_count)

            async def generate():
                api_url = "http://46.167.136.122:11434/api/generate"
                data = {
                    "model": "partai/dorna-llama3",
                    "prompt": prompt
                }
                response_text = ""
                async with client.stream("POST", api_url, json=data) as response:
                    async for line in response.aiter_lines():
                        if not line.strip():
                            continue
                        try:
                            json_line = json.loads(line)
                            token = json_line.get("response", "")
                            if token:
                                cleaned_token = token.replace("<|end_of_text|>", "").replace("</s>", "")
                                response_text += cleaned_token
                                yield cleaned_token
                            if json_line.get("done", False):
                                break
                        except json.JSONDecodeError:
                            continue
                
                # اضافه کردن پاسخ به کش معنایی
                add_to_semantic_cache(user_input, response_text, user_embedding)
                log_interaction(client_id, user_input, response_text)

            return StreamingResponse(generate(), media_type="text/plain")

    except Exception as e:
        logging.exception("❌ خطای کلی: ")
        raise HTTPException(status_code=500, detail=f"خطای داخلی سرور: {str(e)}")

@app.post("/api/chatbot/ask")
async def chatbot_ask(request: QuestionRequest):
    return await ask_endpoint(request)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    client_id = str(uuid.uuid4())

    await websocket.accept()

    try:
        initial_data = await websocket.receive_text()
        try:
            initial_json = json.loads(initial_data)
            if "client_id" in initial_json and initial_json["client_id"]:
                client_id = initial_json["client_id"]
        except json.JSONDecodeError:
            pass
    except WebSocketDisconnect:
        logging.info("کلاینت در ابتدای اتصال قطع شد.")
        return

    if client_id in active_connections:
        await websocket.close(code=4000, reason="این کلاینت قبلاً متصل است")
        return

    active_connections[client_id] = websocket
    logging.info(f"کلاینت {client_id} متصل شد.")
    await websocket.send_text(f"client_id: {client_id}")

    async def keep_alive():
        while True:
            try:
                await asyncio.sleep(30)
                await websocket.ping()
                logging.debug(f"پینگ به کلاینت {client_id} ارسال شد.")
            except Exception as e:
                logging.error(f"خطا در پینگ به کلاینت {client_id}: {str(e)}")
                break

    ping_task = asyncio.create_task(keep_alive())

    try:
        if not hasattr(websocket, "initialized"):
            websocket.initialized = True
            logging.info("مدل و داده‌ها برای اولین بار بارگذاری شد.")

        user_input = ""
        if initial_data:
            try:
                if "question" in initial_json:
                    user_input = initial_json["question"].strip()
            except NameError:
                user_input = initial_data.strip()

        while True:
            if user_input:
                if not user_input:
                    await websocket.send_text("❌ لطفاً سوالی وارد کنید.")
                    user_input = ""
                    continue

                default_response = handle_default_responses(user_input)
                if default_response:
                    await websocket.send_text(default_response)
                    log_interaction(client_id, user_input, default_response)
                    user_input = ""
                    continue

                # استفاده از سؤال اصلی برای جاسازی معنایی
                user_embedding = model.encode(user_input, convert_to_tensor=True)
                cached_answer, cached_question = search_in_semantic_cache(user_input)
                if cached_answer:
                    await websocket.send_text(cached_answer)
                    log_interaction(client_id, user_input, cached_answer)
                    user_input = ""
                    continue

                preprocessed = preprocess_question(user_input)
                best_pair, similarity, _ = get_best_match(preprocessed)

                prompt = generate_prompt_with_history(user_input, best_pair, client_id)

                api_url = "http://46.167.136.122:11434/api/generate"
                data = {"model": "partai/dorna-llama3", "prompt": prompt}
                response_text = ""
                async with client.stream("POST", api_url, json=data) as response:
                    async for line in response.aiter_lines():
                        if not line.strip():
                            continue
                        try:
                            json_line = json.loads(line)
                            token = json_line.get("response", "")
                            if token:
                                cleaned_token = token.replace("<|end_of_text|>", "").replace("</s>", "")
                                response_text += cleaned_token
                                await websocket.send_text(cleaned_token)
                            if json_line.get("done", False):
                                break
                        except json.JSONDecodeError:
                            continue
                
                # اضافه کردن پاسخ به کش معنایی
                add_to_semantic_cache(user_input, response_text, user_embedding)
                log_interaction(client_id, user_input, response_text)
                user_input = ""

            try:
                initial_data = await websocket.receive_text()
                try:
                    initial_json = json.loads(initial_data)
                    if "question" in initial_json:
                        user_input = initial_json["question"].strip()
                    else:
                        user_input = initial_data.strip()
                except json.JSONDecodeError:
                    user_input = initial_data.strip()
            except WebSocketDisconnect:
                logging.info(f"کلاینت {client_id} قطع شد.")
                break

    except Exception as e:
        logging.exception(f"خطا در WebSocket برای کلاینت {client_id}: {str(e)}")
        await websocket.send_text(f"خطایی رخ داده است: {str(e)}")
    finally:
        ping_task.cancel()
        if client_id in active_connections:
            del active_connections[client_id]
            logging.info(f"اتصال کلاینت {client_id} حذف شد.")
        try:
            await websocket.close()
        except:
            pass

class HistoryRequest(BaseModel):
    client_id: str
    count: Optional[int] = None

@app.post("/get_history")
async def get_history_endpoint(request: HistoryRequest):
    try:
        client_id = request.client_id
        count = request.count
        
        if not client_id:
            raise HTTPException(status_code=400, detail="شناسه کاربر ارائه نشده است.")
            
        if client_id not in user_histories:
            return {"client_id": client_id, "history": []}
            
        history = user_histories[client_id].get_history(count)
        return {"client_id": client_id, "history": history}
        
    except Exception as e:
        logging.exception("❌ خطا در دریافت تاریخچه: ")
        raise HTTPException(status_code=500, detail=f"خطای داخلی سرور: {str(e)}")

@app.on_event("shutdown")
async def shutdown_event():
    await client.aclose()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("bot_two:app", host="0.0.0.0", port=8000, reload=True)