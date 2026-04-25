import base64
import io
from typing import List, Dict, Any

import fitz  # PyMuPDF
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
    if image.ndim == 2:
        pil_img = Image.fromarray(image)
    elif image.shape[2] == 4:
        pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA))
    else:
        pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    buffer = io.BytesIO()
    pil_img.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"

def limpiar_fondo_blanco(crop: np.ndarray) -> np.ndarray:
    if crop.shape[2] == 3:
        bgra = cv2.cvtColor(crop, cv2.COLOR_BGR2BGRA)
    else:
        bgra = crop.copy()

    b, g, r, a = cv2.split(bgra)
    # Tolerancia para eliminar el fondo blanco sin tocar colores claros
    white_mask = (r > 240) & (g > 240) & (b > 240)
    a[white_mask] = 0

    return cv2.merge([b, g, r, a])

def color_dominante(crop_bgra: np.ndarray) -> str:
    if crop_bgra.shape[2] != 4:
        return "#000000"

    b, g, r, a = cv2.split(crop_bgra)
    mask = a > 0

    if np.count_nonzero(mask) == 0:
        return "#000000"

    r_avg = int(np.mean(r[mask]))
    g_avg = int(np.mean(g[mask]))
    b_avg = int(np.mean(b[mask]))

    return f"#{r_avg:02X}{g_avg:02X}{b_avg:02X}"

def render_pdf_page(pdf_bytes: bytes, page_number: int, scale: float):
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    if page_number < 0 or page_number >= len(doc):
        page_number = 0

    page = doc[page_number]
    matrix = fitz.Matrix(scale, scale)
    pix = page.get_pixmap(matrix=matrix, alpha=False)

    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
        pix.height, pix.width, pix.n
    )

    if pix.n == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    elif pix.n == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    return doc, page, img, scale

# --- LA NUEVA LÓGICA TIPO CANVA (OBJETOS NATIVOS) ---
def extraer_objetos_pdf(page, image_bgr, scale: float):
    piezas = []
    idx = 0
    
    # 1. EXTRACCIÓN DE TEXTO MEDIANTE OBJETOS DEL PDF
    # Lee exactamente cómo se diseñó el archivo, letra por letra
    data = page.get_text("rawdict")
    for block in data.get("blocks", []):
        if block.get("type") == 0:  # Tipo 0 = Texto
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    for char in span.get("chars", []):
                        ch = char.get("c", "")
                        bbox = char.get("bbox")
                        
                        # Ignorar espacios en blanco o nulos
                        if not ch.strip() or not bbox:
                            continue
                            
                        x0, y0, x1, y1 = bbox
                        
                        # Añadir un mini margen para no cortar los bordes de la fuente
                        margen = 2 / scale
                        x0 -= margen
                        y0 -= margen
                        x1 += margen
                        y1 += margen
                        
                        px0 = int(x0 * scale)
                        py0 = int(y0 * scale)
                        px1 = int(x1 * scale)
                        py1 = int(y1 * scale)
                        
                        # Asegurar que no nos salimos de la imagen
                        px0 = max(0, px0)
                        py0 = max(0, py0)
                        px1 = min(image_bgr.shape[1], px1)
                        py1 = min(image_bgr.shape[0], py1)
                        
                        w = px1 - px0
                        h = py1 - py0
                        
                        if w < 5 or h < 5: 
                            continue

                        # Recortar usando las coordenadas exactas del objeto PDF
                        crop = image_bgr[py0:py1, px0:px1].copy()
                        crop_transparent = limpiar_fondo_blanco(crop)
                        
                        # Si el recorte resultó vacío, lo ignoramos
                        if np.count_nonzero(crop_transparent[:, :, 3]) == 0:
                            continue

                        piezas.append({
                            "id": f"pieza_obj_{idx}",
                            "src": image_to_base64_png(crop_transparent),
                            "char": ch,
                            "confidence": "pdf_object",
                            "x": px0,
                            "y": py0,
                            "w": w,
                            "h": h,
                            "propor": float(w / h) if h else 1,
                            "fila": 0,
                            "color": color_dominante(crop_transparent),
                            "partes": 1
                        })
                        idx += 1

    # 2. PLAN B (VECTORES / CURVAS / IMÁGENES)
    # Si subiste un diseño donde las letras están convertidas a vectores (outlines)
    # o hay logos, usamos una máscara para buscar todo lo que NO fue detectado en el paso 1.
    
    # Crear máscara tapando el texto que ya encontramos
    mask_cv = np.ones(image_bgr.shape[:2], dtype=np.uint8) * 255
    for p in piezas:
        mask_cv[p["y"]:p["y"]+p["h"], p["x"]:p["x"]+p["w"]] = 0
        
    lower_white = np.array([230, 230, 230], dtype=np.uint8)
    upper_white = np.array([255, 255, 255], dtype=np.uint8)
    mask_white = cv2.inRange(image_bgr, lower_white, upper_white)
    thresh = cv2.bitwise_not(mask_white)
    
    # Filtrar solo las áreas que no son texto
    thresh = cv2.bitwise_and(thresh, thresh, mask=mask_cv)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        x, y, w, h = cv2.boundingRect(cnt)
        
        # Tamaño mínimo de objetos vectoriales a recuperar
        if area > 15 and w >= 5 and h >= 5:
            margen = 4
            x0 = max(0, x - margen)
            y0 = max(0, y - margen)
            x1 = min(image_bgr.shape[1], x + w + margen)
            y1 = min(image_bgr.shape[0], y + h + margen)
            
            crop = image_bgr[y0:y1, x0:x1].copy()
            crop_transparent = limpiar_fondo_blanco(crop)
            
            if np.count_nonzero(crop_transparent[:, :, 3]) > 0:
                piezas.append({
                    "id": f"pieza_obj_{idx}",
                    "src": image_to_base64_png(crop_transparent),
                    "char": "?",  # No sabemos qué letra es porque es un vector/imagen
                    "confidence": "vector_fallback",
                    "x": int(x0),
                    "y": int(y0),
                    "w": int(x1 - x0),
                    "h": int(y1 - y0),
                    "propor": float((x1 - x0) / (y1 - y0)) if (y1 - y0) else 1,
                    "fila": 0,
                    "color": color_dominante(crop_transparent),
                    "partes": 1
                })
                idx += 1
                
    # Ordenar las piezas como se leen (arriba hacia abajo, izquierda a derecha)
    piezas.sort(key=lambda p: (round(p["y"] / 40) * 40, p["x"]))
    
    return piezas

@app.get("/api/extract")
def home():
    return {
        "ok": True,
        "service": "Gang Sheet Studio Extractor",
        "endpoint": "/api/extract"
    }

@app.post("/api/extract")
async def extract_pdf(
    file: UploadFile = File(...),
    page: int = Form(0),
    scale: float = Form(3.0)
):
    try:
        pdf_bytes = await file.read()

        if not pdf_bytes:
            return JSONResponse({
                "ok": False,
                "error": "No se recibió ningún archivo PDF."
            }, status_code=400)

        doc, pdf_page, rendered_image, real_scale = render_pdf_page(
            pdf_bytes, page_number=page, scale=scale
        )

        piezas = extraer_objetos_pdf(pdf_page, rendered_image, real_scale)

        return JSONResponse({
            "ok": True,
            "page": page,
            "scale": scale,
            "total_pieces": len(piezas),
            "pieces": piezas
        })

    except Exception as e:
        return JSONResponse({
            "ok": False,
            "error": str(e)
        }, status_code=500)
