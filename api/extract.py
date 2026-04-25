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

    for block in data.get("blocks", []):
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                text = span.get("text", "")
                bbox = span.get("bbox")

                if not text or not bbox:
                    continue

                clean = "".join(
                    ch for ch in text.upper()
                    if ch.isalnum() or ch in "ÁÉÍÓÚÑ"
                )

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


def unir_cajas(cajas: List[Dict[str, Any]]) -> Dict[str, Any]:
    x0 = min(c["x"] for c in cajas)
    y0 = min(c["y"] for c in cajas)
    x1 = max(c["x"] + c["w"] for c in cajas)
    y1 = max(c["y"] + c["h"] for c in cajas)

    return {
        "x": x0,
        "y": y0,
        "w": x1 - x0,
        "h": y1 - y0,
        "partes": cajas
    }


def agrupar_por_filas(cajas: List[Dict[str, Any]], tolerancia_y: int = 45):
    cajas = sorted(cajas, key=lambda c: c["y"])
    filas = []

    for caja in cajas:
        centro_y = caja["y"] + caja["h"] / 2
        mejor_fila = None

        for fila in filas:
            y_min = min(c["y"] for c in fila)
            y_max = max(c["y"] + c["h"] for c in fila)

            if y_min - tolerancia_y <= centro_y <= y_max + tolerancia_y:
                mejor_fila = fila
                break

        if mejor_fila is not None:
            mejor_fila.append(caja)
        else:
            filas.append([caja])

    for fila in filas:
        fila.sort(key=lambda c: c["x"])

    return filas


def agrupar_caracteres_por_fila(fila: List[Dict[str, Any]]):
    if not fila:
        return []

    grupos = []
    grupo_actual = [fila[0]]

    for actual in fila[1:]:
        caja_grupo = unir_cajas(grupo_actual)

        distancia_x = actual["x"] - (caja_grupo["x"] + caja_grupo["w"])
        altura_base = max(caja_grupo["h"], actual["h"])

        se_tocan_verticalmente = (
            actual["y"] < caja_grupo["y"] + caja_grupo["h"]
            and actual["y"] + actual["h"] > caja_grupo["y"]
        )

        cerca_horizontal = distancia_x < altura_base * 0.28

        pequeno_encima = (
            actual["h"] < altura_base * 0.35
            and abs(
                (actual["x"] + actual["w"] / 2)
                - (caja_grupo["x"] + caja_grupo["w"] / 2)
            ) < altura_base * 0.5
        )

        if (
            distancia_x < 0
            or cerca_horizontal
            or pequeno_encima
            or (se_tocan_verticalmente and distancia_x < altura_base * 0.18)
        ):
            grupo_actual.append(actual)
        else:
            grupos.append(unir_cajas(grupo_actual))
            grupo_actual = [actual]

    grupos.append(unir_cajas(grupo_actual))
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
    min_area: int = 80
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

        if area > min_area and w > 4 and h > 8:
            cajas.append({
                "x": int(x),
                "y": int(y),
                "w": int(w),
                "h": int(h),
                "area": float(area)
            })

    filas = agrupar_por_filas(cajas)

    grupos_finales = []

    for index_fila, fila in enumerate(filas):
        grupos = agrupar_caracteres_por_fila(fila)

        for grupo in grupos:
            grupo["fila"] = index_fila
            grupos_finales.append(grupo)

    grupos_finales = [
        g for g in grupos_finales
        if g["w"] > 10 and g["h"] > 15
    ]

    grupos_finales.sort(key=lambda g: (g["y"], g["x"]))

    piezas = []

    for idx, grupo in enumerate(grupos_finales):
        margen = 8

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
            "fila": int(grupo["fila"]),
            "color": color_dominante(crop_transparent),
            "partes": len(grupo.get("partes", []))
        })

    return piezas

# -- CORRECCIÓN APLICADA AQUÍ --
# Cambiamos "/" por "/api/extract" para que coincida con la ruta de Vercel

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
            min_area=80
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
