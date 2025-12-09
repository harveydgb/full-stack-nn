# main.py
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="Signal Lab API",
    description="Endpoints for FFT demos (sin-cos combos) and simple items CRUD.",
    version="0.1.0",
    contact={"name": "Boris Bolliet"},
    license_info={"name": "MIT"},
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # your Next.js dev URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Item(BaseModel):
    name: str
    price: float

@app.get("/")
def hello():
    return {"message": "Hello 👋"}

@app.post("/items")
def create(item: Item):
    return {"ok": True, "item": item}


@app.get("/fft")
def fft_endpoint(n: int = 4096, L: float = 2*np.pi, k: int = 0):
    """
    Compute FFT of sin(5x)*cos(9x) on [0,L) with N samples and return the value at harmonic k.
    - n: number of samples
    - L: domain length
    - k: harmonic index (e.g., 0, ±4, ±14). Maps to FFT bin m ≈ k*L/(2π).
    """
    try:
        return fft_at_k(k=k, n=n, L=L)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))