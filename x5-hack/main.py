import os
import asyncio
from typing import List, Dict, Tuple, Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from ml_service import NERPredictor

MODEL_DIR = os.getenv("MODEL_DIR", "./models/fine_tuned_robert_ner")
MAX_BATCH_SIZE = int(os.getenv("MAX_BATCH_SIZE", "32"))
BATCH_TIMEOUT = float(os.getenv("BATCH_TIMEOUT", "0.01"))
INFERENCE_TIMEOUT = float(os.getenv("INFERENCE_TIMEOUT", "10.0"))

app = FastAPI()

class PredictionRequest(BaseModel):
    input: str

class PredictionResponse(BaseModel):
    start_index: int
    end_index: int
    entity: str

ner = NERPredictor(MODEL_DIR)

_batch_queue: asyncio.Queue = asyncio.Queue()
_batch_worker_started = False

def _batch_predict_sync(texts: List[str]) -> List[List[Dict[str, Any]]]:
    """Синхронный вызов предикта для списка текстов — выполняется в executor."""
    out = []
    for t in texts:
        raw = ner.predict(t)
        formatted = []
        for item in raw:
            if len(item) == 3:
                s, e, lbl = item
                formatted.append({"start_index": int(s), "end_index": int(e), "entity": str(lbl)})
        out.append(formatted)
    return out

async def _batch_worker():
    while True:
        item = await _batch_queue.get()
        texts = [item[0]]
        futures = [item[1]]

        try:
            for _ in range(MAX_BATCH_SIZE - 1):
                other = _batch_queue.get_nowait()
                texts.append(other[0])
                futures.append(other[1])
        except asyncio.QueueEmpty:
            pass

        if len(texts) == 1:
            await asyncio.sleep(BATCH_TIMEOUT)
            try:
                while len(texts) < MAX_BATCH_SIZE:
                    other = _batch_queue.get_nowait()
                    texts.append(other[0])
                    futures.append(other[1])
            except asyncio.QueueEmpty:
                pass

        loop = asyncio.get_running_loop()
        try:
            results = await loop.run_in_executor(None, _batch_predict_sync, texts)
        except Exception as e:
            for f in futures:
                if not f.done():
                    f.set_exception(e)
            continue

        for f, r in zip(futures, results):
            if not f.done():
                f.set_result(r)

@app.on_event("startup")
async def startup_event():
    global _batch_worker_started
    if not _batch_worker_started:
        asyncio.create_task(_batch_worker())
        _batch_worker_started = True

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/api/predict", response_model=List[PredictionResponse])
async def predict(req: PredictionRequest):
    text = req.input or ""
    if not text.strip():
        return []

    loop = asyncio.get_running_loop()
    fut = loop.create_future()
    await _batch_queue.put((text, fut))
    try:
        result = await asyncio.wait_for(fut, timeout=INFERENCE_TIMEOUT)
        return result
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Inference timeout")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")), log_level="info")
