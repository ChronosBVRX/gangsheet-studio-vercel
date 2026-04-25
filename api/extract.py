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
    """Analiza los 2 píxeles más externos del recorte para saber qué color es el fondo y borrarlo."""
    bgra = crop.copy()
    h, w = bgra.shape[:2]
    if h < 4 or w < 4: return bgra

    # Capturar el marco perimetral
    top = bgra[0:2, :].reshape(-1, 4)
    bottom = bgra[h-2:h, :].reshape(-1, 4)
    left = bgra[:, 0:2].reshape(-1, 4)
    right = bgra[:, w-2:w].reshape(-1, 4)
    
    perimetro = np.concatenate([top, bottom, left, right])
    opacos = perimetro[perimetro[:, 3] > 0]
    
    # Si el perímetro ya es transparente, no tocamos nada
    if len(opacos) < len(perimetro) * 0.1: 
        return bgra

    # Encontrar el color que más se repite en el borde
    colores, cuentas = np.unique(opacos[:, :3], axis=0, return_counts=True)
    color_fondo = colores[np.argmax(cuentas)]

    # Si el color dominante no es claro, lo dejamos quieto
    if np.max(cuentas) < len(opacos) * 0.3:
        return bgra

    tolerancia = 15
    lower = np.clip(color_fondo.astype(int) - tolerancia, 0, 255).astype(np.uint8)
    upper = np.clip(color_fondo.astype(int) + tolerancia, 0, 255).astype(np.uint8)

    mask_fondo = cv2.inRange(bgra[:, :, :3], lower, upper)
    
    # Borrar fondo solo si no destruye la letra completa
    if np.count_nonzero(mask_fondo) < h * w * 0.95:
        bgra[mask_fondo > 0, 3] = 0
            
    return bgra

def contar_hijos(i: int, hierarchy: np.ndarray) -> int:
    """Cuenta cuántos elementos hay dentro de un contorno"""
    count = 0
    child = hierarchy[i][2]
    while child != -1:
        count += 1
        child = hierarchy[child][0]
    return count

def merge_boxes(b1, b2):
    return [min(b1[0], b2[0]), min(b1[1], b2[1]), max(b1[2], b2[2]), max(b1[3], b2[3])]

def procesar_vision(page, scale):
    # 1. Renderizamos la imagen visual (WYSIWYG - Lo que ves es lo que hay)
    pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=True)
    img_bgra = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    if pix.n == 3: img_bgra = cv2.cvtColor(img_bgra, cv2.COLOR_RGB2BGRA)
    else: img_bgra = cv2.cvtColor(img_bgra, cv2.COLOR_RGBA2BGRA)

    # 2. Detección de Contornos (Canny) - Ignora el color, busca solo las formas
    gray = cv2.cvtColor(img_bgra, cv2.COLOR_BGRA2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(blurred, 30, 150)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    contours, hierarchy = cv2.findContours(closed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    cajas = []
    if hierarchy is not None:
        hierarchy = hierarchy[0]
        for i, cnt in enumerate(contours):
            x, y, w, h = cv2.boundingRect(cnt)
            area = w * h
            
            # Filtrar basura microscópica
            if area < 20 or w < 4 or h < 4: continue
            
            # MAGIA: Si tiene más de 3 hijos (como un banner), lo saltamos.
            hijos = contar_hijos(i, hierarchy)
            if hijos > 3: continue 
            
            # MAGIA: Si está dentro de una letra válida (ej. el hueco de la O), lo saltamos.
            padre = hierarchy[i][3]
            if padre != -1:
                hijos_padre = contar_hijos(padre, hierarchy)
                if hijos_padre <= 3: continue 
                
            cajas.append({"bbox": [x, y, x+w, y+h]})

    # 3. Agrupar piezas dependientes (El punto de la 'i' o la tilde de la 'Ñ')
    grupos = []
    for caja in cajas:
        matched = False
        for g in grupos:
            b1 = g["bbox"]
            b2 = caja["bbox"]
            
            x_overlap = min(b1[2], b2[2]) - max(b1[0], b2[0])
            if x_overlap > -2:  # Están en la misma columna vertical
                h1 = b1[3] - b1[1]
                h2 = b2[3] - b2[1]
                alt_peq = min(h1, h2)
                alt_gran = max(h1, h2)
                y_dist = min(abs(b1[1] - b2[3]), abs(b2[1] - b1[3]))
                
                # Solo agrupar si uno es mucho más chico (tilde/punto) y están muy cerca
                if (alt_peq < alt_gran * 0.5) and (y_dist < alt_peq * 2.0):
                    g["bbox"] = merge_boxes(b1, b2)
                    matched = True
                    break
        if not matched:
            grupos.append(caja)

    # 4. Leer texto nativo solo para sugerir letras (si el PDF no tiene las letras en curvas)
    textos_nativos = []
    for block in page.get_text("dict").get("blocks", []):
        if block.get("type") == 0:
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    for char in span.get("chars", []):
                        ch = char.get("c", "")
                        bbox = char.get("bbox")
                        if ch.strip() and bbox:
                            textos_nativos.append({
                                "char": ch,
                                "cx": (bbox[0]+bbox[2])*scale/2,
                                "cy": (bbox[1]+bbox[3])*scale/2
                            })

    piezas = []
    idx = 0
    for g in grupos:
        x0, y0, x1, y1 = g["bbox"]
        
        # Le damos un margen para que el Limpiador de Fondo tenga de dónde "leer" el color perimetral
        margen = 5
        x0 = max(0, x0 - margen)
        y0 = max(0, y0 - margen)
        x1 = min(img_bgra.shape[1], x1 + margen)
        y1 = min(img_bgra.shape[0], y1 + margen)
        
        crop = img_bgra[y0:y1, x0:x1].copy()
        crop_clean = limpiar_fondo_inteligente(crop)
        
        if np.count_nonzero(crop_clean[:, :, 3]) == 0: continue
        
        cx = (x0 + x1) / 2
        cy = (y0 + y1) / 2
        
        # Asignar la letra correspondiente si se puede (si no, queda en "?")
        char_sugerido = "?"
        mejor_dist = float('inf')
        mejor_t_idx = -1
        
        for i_t, t in enumerate(textos_nativos):
            dist = ((cx - t["cx"])**2 + (cy - t["cy"])**2)**0.5
            if dist < min(x1-x0, y1-y0) * 0.8 and dist < mejor_dist:
                mejor_dist = dist
                mejor_t_idx = i_t
                
        if mejor_t_idx != -1:
            char_sugerido = textos_nativos[mejor_t_idx]["char"]
            textos_nativos.pop(mejor_t_idx)  # Quitamos la letra usada

        piezas.append({
            "id": f"pieza_{idx}",
            "src": image_to_base64_png(crop_clean),
            "char": char_sugerido,
            "x": x0, "y": y0, "w": x1-x0, "h": y1-y0,
            "propor": (x1-x0)/(y1-y0) if (y1-y0) else 1,
            "fila": 0,
            "color": color_dominante(crop_clean)
        })
        idx += 1

    # Ordenar izquierda a derecha, arriba a abajo
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

        piezas = procesar_vision(pdf_page, scale)

        return JSONResponse({
            "ok": True,
            "page": page,
            "scale": scale,
            "total_pieces": len(piezas),
            "pieces": piezas
        })

    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)
