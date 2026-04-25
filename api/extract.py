import base64
import io
from typing import List, Dict, Any

import fitz
import cv2
import numpy as np
from PIL import Image
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

app = FastAPI(title="Gang Sheet Studio Extractor")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def image_to_base64_png(image: np.ndarray) -> str:
    pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA))
    buffer = io.BytesIO()
    pil_img.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"

def color_dominante(crop_bgra: np.ndarray) -> str:
    b, g, r, a = cv2.split(crop_bgra)
    mask = a > 0
    if np.count_nonzero(mask) == 0: 
        return "#000000"
    return f"#{int(np.mean(r[mask])):02X}{int(np.mean(g[mask])):02X}{int(np.mean(b[mask])):02X}"

def limpiar_fondo_inteligente(crop: np.ndarray) -> np.ndarray:
    """Lee el perímetro de la imagen para borrar inteligentemente el fondo sin tocar la letra."""
    bgra = crop.copy()
    h, w = bgra.shape[:2]
    if h < 4 or w < 4: return bgra

    # Leer todos los píxeles del contorno exterior
    perimetro = np.concatenate([bgra[0, :], bgra[-1, :], bgra[:, 0], bgra[:, -1]])
    
    opacos = perimetro[perimetro[:, 3] > 0]
    # Si menos del 20% del perímetro es opaco, significa que ya está transparente.
    if len(opacos) < (h + w) * 0.2: 
        return bgra

    colores, cuentas = np.unique(opacos[:, :3], axis=0, return_counts=True)
    color_fondo = colores[np.argmax(cuentas)]

    # Si el color predominante no representa una buena parte del borde, no es un fondo sólido
    if np.max(cuentas) < len(opacos) * 0.3:
        return bgra

    tolerancia = 15
    lower = np.clip(color_fondo.astype(int) - tolerancia, 0, 255).astype(np.uint8)
    upper = np.clip(color_fondo.astype(int) + tolerancia, 0, 255).astype(np.uint8)

    mask_fondo = cv2.inRange(bgra[:, :, :3], lower, upper)
    
    # Prevención: Si resulta que toda la imagen es de ese color, no la borramos completa
    if np.count_nonzero(mask_fondo) == h * w:
        return bgra
        
    bgra[mask_fondo > 0, 3] = 0
    return bgra

def calculate_overlap(b1, b2):
    """Calcula qué tanto están apilados dos objetos (ideal para unir bordes y rellenos)"""
    x_left = max(b1[0], b2[0])
    y_top = max(b1[1], b2[1])
    x_right = min(b1[2], b2[2])
    y_bottom = min(b1[3], b2[3])
    
    if x_right <= x_left or y_bottom <= y_top: return 0.0
    
    inter = (x_right - x_left) * (y_bottom - y_top)
    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    
    return inter / min(a1, a2) if min(a1, a2) > 0 else 0

def merge_boxes(b1, b2):
    return [min(b1[0], b2[0]), min(b1[1], b2[1]), max(b1[2], b2[2]), max(b1[3], b2[3])]

def agrupar_vectores(vectores):
    """Agrupa vectores SOLO si están apilados como capas o si son modificadores (como puntos o tildes)."""
    grupos = []
    for v in vectores:
        matched = False
        for g in grupos:
            overlap = calculate_overlap(g["bbox"], v["bbox"])
            
            # Detectar modificadores (puntos sobre la i, acentos)
            b1, b2 = g["bbox"], v["bbox"]
            x_overlap = min(b1[2], b2[2]) - max(b1[0], b2[0])
            y_dist = min(abs(b1[1] - b2[3]), abs(b2[1] - b1[3]))
            alt_min = min(b1[3]-b1[1], b2[3]-b2[1])
            
            is_mod = (x_overlap > 0) and (y_dist < alt_min * 1.5) and (b1[3]<b2[1] or b2[3]<b1[1])

            # Solo fusionar si hay más del 15% de overlap (están uno sobre otro) o si es un acento
            if overlap > 0.15 or is_mod:
                g["bbox"] = merge_boxes(g["bbox"], v["bbox"])
                matched = True
                break
        if not matched:
            grupos.append(v)
            
    return grupos

def extraer_objetos_pdf_nativos(page, scale: float):
    pix_alpha = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=True)
    img_bgra = np.frombuffer(pix_alpha.samples, dtype=np.uint8).reshape(pix_alpha.height, pix_alpha.width, pix_alpha.n)
    
    if pix_alpha.n == 3: img_bgra = cv2.cvtColor(img_bgra, cv2.COLOR_RGB2BGRA)
    else: img_bgra = cv2.cvtColor(img_bgra, cv2.COLOR_RGBA2BGRA)

    page_w = page.rect.width * scale
    page_h = page.rect.height * scale

    textos = []
    vectores = []

    # 1. LEER TEXTOS NATIVOS
    data = page.get_text("rawdict")
    allowed_chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789ÁÉÍÓÚáéíóúÑñ.,;:-_!?'\"()[]{}@*/\\&#%+=$€"
    
    for block in data.get("blocks", []):
        if block.get("type") == 0:
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    for char in span.get("chars", []):
                        ch = char.get("c", "")
                        bbox = char.get("bbox")
                        if ch in allowed_chars and bbox:
                            textos.append({
                                "char": ch,
                                "bbox": [bbox[0]*scale, bbox[1]*scale, bbox[2]*scale, bbox[3]*scale]
                            })

    # 2. LEER OBJETOS VECTORIALES (Eliminando fondos gigantes)
    for path in page.get_drawings():
        bbox = path["rect"]
        w = bbox.width * scale
        h = bbox.height * scale
        
        # Filtro: Si abarca más del 40% del ancho o alto de la página, es un banner/fondo. ¡Ignorarlo!
        if w > page_w * 0.4 or h > page_h * 0.4: continue
        
        if w > 3 and h > 3:
            vectores.append({
                "char": "?",
                "bbox": [bbox.x0*scale, bbox.y0*scale, bbox.x1*scale, bbox.y1*scale]
            })

    # 3. FILTRAR DUPLICADOS Y AGRUPAR VECTORES
    vectores_filtrados = []
    for v in vectores:
        es_duplicado = False
        for t in textos:
            # Si un vector envuelve exactamente a un texto, es su contorno duplicado. Lo omitimos.
            if calculate_overlap(v["bbox"], t["bbox"]) > 0.5:
                es_duplicado = True
                break
        if not es_duplicado:
            vectores_filtrados.append(v)

    grupos_vectores = agrupar_vectores(vectores_filtrados)
    
    # Juntar todo (Textos individuales + Vectores Agrupados)
    todas_las_piezas = textos + grupos_vectores

    piezas_finales = []
    idx = 0
    
    for obj in todas_las_piezas:
        x0, y0, x1, y1 = obj["bbox"]
        
        # Margen de protección para que el limpiador no toque la letra real
        margen = 4
        x0 = max(0, int(x0) - margen)
        y0 = max(0, int(y0) - margen)
        x1 = min(img_bgra.shape[1], int(x1) + margen)
        y1 = min(img_bgra.shape[0], int(y1) + margen)
        
        w, h = x1 - x0, y1 - y0
        if w < 5 or h < 5: continue
        
        crop = img_bgra[y0:y1, x0:x1].copy()
        crop_clean = limpiar_fondo_inteligente(crop)
        
        # Si quedó vacío tras limpiar, descartar
        if np.count_nonzero(crop_clean[:, :, 3]) == 0: continue
        
        piezas_finales.append({
            "id": f"pieza_{idx}",
            "src": image_to_base64_png(crop_clean),
            "char": obj["char"],
            "confidence": "pdf_object",
            "x": x0, "y": y0, "w": w, "h": h,
            "propor": w/h if h else 1,
            "fila": 0,
            "color": color_dominante(crop_clean),
            "partes": 1
        })
        idx += 1

    # Ordenar lectura: arriba hacia abajo, izquierda a derecha
    piezas_finales.sort(key=lambda p: (round(p["y"] / 40) * 40, p["x"]))
    return piezas_finales

@app.get("/api/extract")
def home():
    return {"ok": True, "service": "Gang Sheet Studio Extractor", "endpoint": "/api/extract"}

@app.post("/api/extract")
async def extract_pdf(
    file: UploadFile = File(...),
    page: int = Form(0),
    scale: float = Form(3.0)
):
    try:
        pdf_bytes = await file.read()
        if not pdf_bytes:
            return JSONResponse({"ok": False, "error": "No se recibió archivo PDF."}, status_code=400)

        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        if page < 0 or page >= len(doc): page = 0
        pdf_page = doc[page]

        piezas = extraer_objetos_pdf_nativos(pdf_page, scale)

        return JSONResponse({
            "ok": True,
            "page": page,
            "scale": scale,
            "total_pieces": len(piezas),
            "pieces": piezas
        })

    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)
