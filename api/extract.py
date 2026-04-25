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


def render_pdf_page(pdf_bytes: bytes, page_number: int = 0, scale: float = 3.0):
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")

    if page_number < 0 or page_number >= len(doc):
        page_number = 0

    page = doc[page_number]
    matrix = fitz.Matrix(scale, scale)
    pix = page.get_pixmap(matrix=matrix, alpha=False)

    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
        pix.height,
        pix.width,
        pix.n
    )

    if pix.n == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    elif pix.n == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    return doc, page, img, scale


def extract_pdf_text_positions(page, scale: float) -> List[Dict[str, Any]]:
    chars = []
    data = page.get_text("dict")

    # Ampliamos los caracteres permitidos para incluir símbolos, puntuación y acentos
    allowed_chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789ÁÉÍÓÚáéíóúÑñ.,;:-_!?'\"()[]{}@*/\\&#%+=$€"

    for block in data.get("blocks", []):
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                text = span.get("text", "")
                bbox = span.get("bbox")

                if not text or not bbox:
                    continue

                # Filtrar manteniendo los caracteres válidos
                clean = "".join(ch for ch in text if ch in allowed_chars)

                if not clean:
                    continue

                x0, y0, x1, y1 = bbox
                width = (x1 - x0) * scale
                height = (y1 - y0) * scale
                char_width = width / max(len(clean), 1)

                for i, ch in enumerate(clean):
                    cx = (x0 * scale) + (char_width * i) + (char_width / 2)
                    cy = (y0 * scale) + (height / 2)

                    chars.append({
                        "char": ch,
                        "cx": float(cx),
                        "cy": float(cy),
                        "source": "pdf_text"
                    })

    return chars


def limpiar_fondo_blanco(crop: np.ndarray) -> np.ndarray:
    if crop.shape[2] == 3:
        bgra = cv2.cvtColor(crop, cv2.COLOR_BGR2BGRA)
    else:
        bgra = crop.copy()

    b, g, r, a = cv2.split(bgra)

    white_mask = (r > 235) & (g > 235) & (b > 235)
    a[white_mask] = 0

    return cv2.merge([b, g, r, a])


# NUEVA FUNCIÓN: Agrupar solo partes del MISMO carácter (como la 'i' y su punto, o la 'Ñ' y su tilde)
def agrupar_partes_caracter(cajas: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not cajas:
        return []

    # Ordenar de izquierda a derecha
    cajas_ordenadas = sorted(cajas, key=lambda c: c["x"])
    grupos = []

    for caja in cajas_ordenadas:
        agregado = False
        caja_centro_x = caja["x"] + caja["w"] / 2

        for grupo in grupos:
            grupo_min_x = grupo["x"]
            grupo_max_x = grupo["x"] + grupo["w"]

            # Si el centro de la nueva caja cae dentro del ancho del grupo existente
            # (con un pequeño margen de tolerancia), pertenecen a la misma letra (ej. acento arriba)
            if (grupo_min_x - 5) <= caja_centro_x <= (grupo_max_x + 5):
                grupo["partes"].append(caja)

                # Expandir los límites del grupo para abarcar la nueva parte
                nuevo_x = min(grupo["x"], caja["x"])
                nuevo_y = min(grupo["y"], caja["y"])
                nuevo_max_x = max(grupo["x"] + grupo["w"], caja["x"] + caja["w"])
                nuevo_max_y = max(grupo["y"] + grupo["h"], caja["y"] + caja["h"])

                grupo["x"] = nuevo_x
                grupo["y"] = nuevo_y
                grupo["w"] = nuevo_max_x - nuevo_x
                grupo["h"] = nuevo_max_y - nuevo_y

                agregado = True
                break

        if not agregado:
            # Crear un nuevo grupo de carácter individual
            grupos.append({
                "x": caja["x"],
                "y": caja["y"],
                "w": caja["w"],
                "h": caja["h"],
                "partes": [caja]
            })

    return grupos


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


def buscar_match_pdf(grupo: Dict[str, Any], elementos_pdf: List[Dict[str, Any]]):
    if not elementos_pdf:
        return ""

    cx = grupo["x"] + grupo["w"] / 2
    cy = grupo["y"] + grupo["h"] / 2

    mejor_idx = -1
    mejor_distancia = float("inf")

    for i, item in enumerate(elementos_pdf):
        dist = ((cx - item["cx"]) ** 2 + (cy - item["cy"]) ** 2) ** 0.5

        if dist < mejor_distancia:
            mejor_distancia = dist
            mejor_idx = i

    tolerancia = max(grupo["w"], grupo["h"]) * 0.75

    if mejor_idx != -1 and mejor_distancia < tolerancia:
        char = elementos_pdf[mejor_idx]["char"]
        elementos_pdf.pop(mejor_idx)
        return char

    return ""


def detectar_piezas(
    image_bgr: np.ndarray,
    elementos_pdf: List[Dict[str, Any]],
    min_area: int = 15  # Reducido drásticamente para atrapar puntos y comas
):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    _, thresh = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    contours, _ = cv2.findContours(
        thresh,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    cajas = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        x, y, w, h = cv2.boundingRect(cnt)

        # Filtros relajados: Si mide al menos 2x2 píxeles, lo procesamos
        if area > min_area and w >= 2 and h >= 2:
            cajas.append({
                "x": int(x),
                "y": int(y),
                "w": int(w),
                "h": int(h),
                "area": float(area)
            })

    # Ahora agrupamos SOLO verticalmente (acentos, tildes, la 'i')
    grupos_finales = agrupar_partes_caracter(cajas)

    # Ordenar lectura de arriba hacia abajo, de izquierda a derecha
    grupos_finales.sort(key=lambda g: (round(g["y"] / 20) * 20, g["x"]))

    piezas = []

    for idx, grupo in enumerate(grupos_finales):
        margen = 4  # Margen más cerrado para no captar basura del vecino

        x = max(0, grupo["x"] - margen)
        y = max(0, grupo["y"] - margen)
        w = min(image_bgr.shape[1] - x, grupo["w"] + margen * 2)
        h = min(image_bgr.shape[0] - y, grupo["h"] + margen * 2)

        crop = image_bgr[y:y + h, x:x + w].copy()
        crop_transparent = limpiar_fondo_blanco(crop)

        base64_png = image_to_base64_png(crop_transparent)
        suggested_char = buscar_match_pdf(grupo, elementos_pdf)

        piezas.append({
            "id": f"pieza_{idx}",
            "src": base64_png,
            "char": suggested_char,
            "confidence": "pdf_match" if suggested_char else "manual",
            "x": int(x),
            "y": int(y),
            "w": int(w),
            "h": int(h),
            "propor": float(w / h) if h else 1,
            "fila": 0,
            "color": color_dominante(crop_transparent),
            "partes": len(grupo.get("partes", []))
        })

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
            pdf_bytes,
            page_number=page,
            scale=scale
        )

        elementos_pdf = extract_pdf_text_positions(pdf_page, real_scale)

        piezas = detectar_piezas(
            rendered_image,
            elementos_pdf=elementos_pdf,
            min_area=15  # Coincide con la relajación de filtros
        )

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
