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
    """Revisa las esquinas de la imagen extraída para borrar el fondo sin dañar los colores internos (como el blanco de Boca Jr)."""
    bgra = crop.copy()
    h, w = bgra.shape[:2]
    if h < 4 or w < 4: 
        return bgra

    # Analizar las 4 esquinas del objeto para detectar el color del fondo (ej. el banner azul)
    esquinas = np.array([
        bgra[0, 0], bgra[0, w-1], 
        bgra[h-1, 0], bgra[h-1, w-1]
    ])
    
    esquinas_opacas = esquinas[esquinas[:, 3] > 0]
    if len(esquinas_opacas) < 2:
        return bgra

    colores, cuentas = np.unique(esquinas_opacas[:, :3], axis=0, return_counts=True)
    color_fondo = colores[np.argmax(cuentas)]

    # Si el color no está presente en al menos 2 esquinas, no es fondo.
    if np.max(cuentas) < 2:
        return bgra

    tolerancia = 20
    lower = np.clip(color_fondo.astype(int) - tolerancia, 0, 255).astype(np.uint8)
    upper = np.clip(color_fondo.astype(int) + tolerancia, 0, 255).astype(np.uint8)

    mask_fondo = cv2.inRange(bgra[:, :, :3], lower, upper)
    
    # Borramos el color detectado (solo el fondo)
    bgra[mask_fondo > 0, 3] = 0

    return bgra

def intersectan(b1, b2, margin=1):
    # Revisa matemáticamente si dos objetos vectoriales se tocan
    if b1[2] + margin < b2[0] or b1[0] - margin > b2[2]: return False
    if b1[3] + margin < b2[1] or b1[1] - margin > b2[3]: return False
    return True

def merge_boxes(b1, b2):
    return [
        min(b1[0], b2[0]),
        min(b1[1], b2[1]),
        max(b1[2], b2[2]),
        max(b1[3], b2[3])
    ]

def agrupar_objetos(objetos):
    # Fusiona capas de diseño (ej. Borde amarillo + Relleno blanco)
    grupos = []
    for obj in objetos:
        matched = False
        for g in grupos:
            if intersectan(g["bbox"], obj["bbox"]):
                g["bbox"] = merge_boxes(g["bbox"], obj["bbox"])
                if g["char"] == "?" and obj["char"] != "?":
                    g["char"] = obj["char"]
                matched = True
                break
        if not matched:
            grupos.append({"char": obj["char"], "bbox": obj["bbox"]})
            
    # Segunda pasada de seguridad para agrupar
    while True:
        merged_any = False
        nuevos_grupos = []
        while grupos:
            actual = grupos.pop(0)
            merged = False
            for i, g in enumerate(nuevos_grupos):
                if intersectan(g["bbox"], actual["bbox"]):
                    nuevos_grupos[i]["bbox"] = merge_boxes(g["bbox"], actual["bbox"])
                    if nuevos_grupos[i]["char"] == "?" and actual["char"] != "?":
                        nuevos_grupos[i]["char"] = actual["char"]
                    merged = True
                    merged_any = True
                    break
            if not merged:
                nuevos_grupos.append(actual)
        grupos = nuevos_grupos
        if not merged_any:
            break
    return grupos

def extraer_objetos_pdf_nativos(page, scale: float):
    # Renderizar página con transparencia
    pix_alpha = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=True)
    img_bgra = np.frombuffer(pix_alpha.samples, dtype=np.uint8).reshape(pix_alpha.height, pix_alpha.width, pix_alpha.n)
    
    if pix_alpha.n == 3:
        img_bgra = cv2.cvtColor(img_bgra, cv2.COLOR_RGB2BGRA)
    else:
        img_bgra = cv2.cvtColor(img_bgra, cv2.COLOR_RGBA2BGRA)

    page_w = page.rect.width * scale
    page_h = page.rect.height * scale
    page_area = page_w * page_h

    objetos = []

    # 1. LEER OBJETOS DE TEXTO
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
                            objetos.append({
                                "char": ch,
                                "bbox": [bbox[0]*scale, bbox[1]*scale, bbox[2]*scale, bbox[3]*scale]
                            })

    # 2. LEER OBJETOS VECTORIALES (Trazos, Curvas, Capas)
    for path in page.get_drawings():
        bbox = path["rect"]
        w = bbox.width * scale
        h = bbox.height * scale
        area = w * h
        
        # FILTRO ANTI-BANNERS: Ignorar objetos rectangulares gigantes
        if area > page_area * 0.20: continue
        if w > page_w * 0.5: continue
        if h > page_h * 0.5: continue
        
        if w > 3 and h > 3:
            objetos.append({
                "char": "?",
                "bbox": [bbox.x0*scale, bbox.y0*scale, bbox.x1*scale, bbox.y1*scale]
            })

    # 3. AGRUPAR CAPAS MATEMÁTICAMENTE
    grupos = agrupar_objetos(objetos)

    piezas = []
    idx = 0
    
    for g in grupos:
        x0, y0, x1, y1 = g["bbox"]
        
        # Margen mínimo para capturar el borde de las letras
        margen = 2
        x0 = max(0, int(x0) - margen)
        y0 = max(0, int(y0) - margen)
        x1 = min(img_bgra.shape[1], int(x1) + margen)
        y1 = min(img_bgra.shape[0], int(y1) + margen)
        
        w = x1 - x0
        h = y1 - y0
        
        if w < 5 or h < 5: continue
        
        crop = img_bgra[y0:y1, x0:x1].copy()
        
        # Extracción final limpia sin destruir colores
        crop_clean = limpiar_fondo_inteligente(crop)
        
        if np.count_nonzero(crop_clean[:, :, 3]) == 0: continue
        
        piezas.append({
            "id": f"pieza_{idx}",
            "src": image_to_base64_png(crop_clean),
            "char": g["char"],
            "confidence": "pdf_object",
            "x": x0, "y": y0, "w": w, "h": h,
            "propor": w/h if h else 1,
            "fila": 0,
            "color": color_dominante(crop_clean),
            "partes": 1
        })
        idx += 1

    # Ordenar lectura: arriba hacia abajo, izquierda a derecha
    piezas.sort(key=lambda p: (round(p["y"] / 40) * 40, p["x"]))
    return piezas

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
