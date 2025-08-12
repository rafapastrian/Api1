# main.py  —  2D Pro (relighting + camera hint + grading)
import os, math
from io import BytesIO
from uuid import uuid4

import numpy as np
from PIL import Image, ImageFilter, ImageOps, ImageEnhance
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from rembg import remove

# --- App & CORS ---
app = FastAPI(title="CR Studio 2D Pro", version="A1")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # en prod limita a tu dominio
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- I/O local simple (puedes cambiar a S3/GCS) ---
UPLOAD_DIR, OUTPUT_DIR = "uploads", "outputs"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
app.mount("/outputs", StaticFiles(directory=OUTPUT_DIR), name="outputs")

# --- Estilos (presets) ---
PRESETS = {
    "clean_premium": dict(
        bg=("champagne_gray",),
        key_dir=( -0.6, -0.3),   # izquierda-arriba
        fill_dir=( 0.7,  0.1),   # derecha
        rim_side="left",
        contrast=1.05, saturation=0.98, tone="slight_film",
        warp_deg=6,
    ),
    "lifestyle_sutil": dict(
        bg=("soft_desaturated", "hint_scene"),
        key_dir=( -0.4, -0.2),
        fill_dir=( 0.6,  0.0),
        rim_side="right",
        contrast=1.02, saturation=1.05, tone="warm_soft",
        warp_deg=5,
    ),
    "tech_cool": dict(
        bg=("cool_teal_beams",),
        key_dir=( -0.7, -0.25),
        fill_dir=( 0.8,  0.05),
        rim_side="left",
        contrast=1.08, saturation=0.95, tone="crisp_cool",
        warp_deg=8,
    ),
}

# ---------- Utilidades visuales ----------
def pil_to_np(img: Image.Image) -> np.ndarray:
    return np.asarray(img).astype(np.float32) / 255.0

def np_to_pil(arr: np.ndarray) -> Image.Image:
    arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)

def make_bg(w, h, preset):
    if "champagne_gray" in preset:
        top = np.array([240, 240, 242]) / 255
        bot = np.array([225, 225, 230]) / 255
        g = np.linspace(0, 1, h)[:, None]
        arr = (top*(1-g) + bot*g)
        arr = np.repeat(arr, w, axis=1)
        return np_to_pil(arr)
    if "cool_teal_beams" in preset:
        # base gris frío con beams suaves diagonales
        base = np.ones((h, w, 3), dtype=np.float32)*0.93
        yy, xx = np.mgrid[0:h, 0:w]
        beam = (np.sin((xx*0.008 + yy*0.004)) + 1)/2 * 0.08
        tint = np.array([0.88, 0.95, 1.00])  # leve azulado
        arr = base*tint + beam[...,None]
        return np_to_pil(arr)
    if "soft_desaturated" in preset:
        top = np.array([238, 238, 238]) / 255
        bot = np.array([228, 230, 232]) / 255
        g = np.linspace(0, 1, h)[:, None]
        arr = (top*(1-g) + bot*g)
        arr = np.repeat(arr, w, axis=1)
        # viñeta muy sutil para “escena”
        img = np_to_pil(arr)
        vign = Image.new("L", (w, h), 0)
        vign_draw = ImageOps.invert(Image.radial_gradient("L"))
        vign = vign_draw.resize((w, h)).filter(ImageFilter.GaussianBlur(max(w,h)//12))
        img = Image.composite(img, ImageEnhance.Brightness(img).enhance(0.96), vign)
        return img
    # fallback
    return Image.new("RGB", (w, h), (245, 245, 247))

def normalize(vx, vy):
    n = math.hypot(vx, vy);  n = 1e-6 if n == 0 else n
    return vx/n, vy/n

def relight_object(obj_rgba: Image.Image, key_dir, fill_dir, rim_side="left"):
    """Aplica mapa de luz suave: key+fill y rim sutil, sin tocar color base."""
    w, h = obj_rgba.size
    a = obj_rgba.split()[-1]  # alpha
    a_np = np.asarray(a).astype(np.float32)/255.0
    matte = a_np > 0.01

    # gradiente direccional
    yy, xx = np.mgrid[0:h, 0:w]
    cx, cy = w/2, h/2
    kx, ky = normalize(*key_dir)
    fx, fy = normalize(*fill_dir)

    # campos de iluminación [0..1] con clamp al área del objeto
    key = ( (xx-cx)*kx + (yy-cy)*ky ) / max(w,h) * 2.2
    fill = ( (xx-cx)*fx + (yy-cy)*fy ) / max(w,h) * 1.6
    key = np.clip(0.5 - key, 0, 1)
    fill = np.clip(0.5 - fill, 0, 1)

    # rim (contraluz)
    edge = Image.fromarray((a_np*255).astype(np.uint8)).filter(ImageFilter.FIND_EDGES)
    edge = np.asarray(edge).astype(np.float32)/255.0
    rim = edge
    if rim_side == "left":
        ramp = np.clip( (xx - cx)/w + 0.5, 0, 1)
        rim = rim * (1 - ramp)
    else:
        ramp = np.clip( (cx - xx)/w + 0.5, 0, 1)
        rim = rim * (1 - ramp)
    rim = (rim * 0.35).astype(np.float32)

    # aplicar en multiplicativo-suave al objeto (sin clipping duro)
    obj = obj_rgba.convert("RGB")
    obj_np = pil_to_np(obj)
    light = (0.85 + 0.25*key[...,None] + 0.15*fill[...,None] + 0.10*rim[...,None])
    out_np = obj_np * (light)
    # reinyectar solo en el área del objeto
    out_np = obj_np*(1-matte[...,None]) + out_np*(matte[...,None])
    out = np_to_pil(out_np)
    out.putalpha(a)
    return out

def perspective_warp(obj_rgba: Image.Image, deg=6):
    """Simula leve cambio de cámara (yaw) con un warp de perspectiva suave."""
    deg = max(-12, min(12, float(deg)))
    w, h = obj_rgba.size
    shift = math.tan(math.radians(deg)) * h * 0.12
    if deg >= 0:
        src = [(0,0),(w,0),(w,h),(0,h)]
        dst = [(0+shift,0),(w,0),(w-shift,h),(0,h)]
    else:
        src = [(0,0),(w,0),(w,h),(0,h)]
        dst = [(0,0),(w-shift,0),(w,h),(0+shift,h)]
    return obj_rgba.transform((w,h), Image.QUAD, data=sum(dst,()), resample=Image.BICUBIC)

def make_shadow(alpha: Image.Image, strength=0.22, blur_ratio=0.02, y_offset_ratio=0.01, squash_ratio=0.88):
    w, h = alpha.size
    a = alpha.convert("L")
    nh = max(1, int(h * squash_ratio))
    shadow_mask = a.resize((w, nh), resample=Image.BICUBIC)
    canvas = Image.new("L", (w, h), 0)
    y_off = int(h * (1 - squash_ratio) + h * y_offset_ratio)
    canvas.paste(shadow_mask, (0, min(max(0, y_off), h - nh)))
    blur_radius = max(1, int(max(w, h) * blur_ratio))
    canvas = canvas.filter(ImageFilter.GaussianBlur(blur_radius))
    shadow = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    canvas = Image.eval(canvas, lambda px: int(px * strength))
    shadow.putalpha(canvas)
    return shadow

def grading(img: Image.Image, contrast=1.0, saturation=1.0, tone="neutral"):
    out = ImageEnhance.Contrast(img).enhance(contrast)
    out = ImageEnhance.Color(out).enhance(saturation)
    if tone == "warm_soft":
        r,g,b = out.split()
        r = ImageEnhance.Brightness(r).enhance(1.02)
        out = Image.merge("RGB", (r,g,b))
    elif tone == "crisp_cool":
        r,g,b = out.split()
        b = ImageEnhance.Brightness(b).enhance(1.03)
        out = Image.merge("RGB", (r,g,b))
    elif tone == "slight_film":
        out = ImageOps.colorize(out.convert("L"), black="#0b0b0b", white="#f3f3f3").convert("RGB").blend(out, 0.85)
    return out

# ---------- Endpoint principal 2D ----------
@app.post("/studio/context2d")
async def studio_context2d(
    file: UploadFile = File(...),
    style: str = Form("clean_premium"),
    angle_hint_deg: float = Form(0)  # -12..12 adicional a preset
):
    style = style.lower().strip()
    if style not in PRESETS:
        return JSONResponse({"error": f"style must be one of {list(PRESETS.keys())}"}, status_code=400)

    try:
        raw = await file.read()
        src = Image.open(BytesIO(raw)).convert("RGBA")
        w, h = src.size

        # 1) Cutout (protege objeto)
        cut = remove(src)  # RGBA con alpha
        alpha = cut.split()[-1]

        # 2) Fondo coherente con estilo
        bg = make_bg(w, h, PRESETS[style]["bg"]).convert("RGBA")

        # 3) “Cámara” simulada: warp sutil
        deg = PRESETS[style]["warp_deg"] + float(angle_hint_deg)
        cut_warp = perspective_warp(cut, deg=deg)

        # 4) Relighting (key+fill+rim) SIN alterar color base global
        cut_light = relight_object(
            cut_warp,
            key_dir=PRESETS[style]["key_dir"],
            fill_dir=PRESETS[style]["fill_dir"],
            rim_side=PRESETS[style]["rim_side"],
        )

        # 5) Sombra suave (en fondo únicamente)
        shadow = make_shadow(alpha.resize((w, h)))

        comp = Image.alpha_composite(bg, shadow)
        comp = Image.alpha_composite(comp, cut_light)

        # 6) Color grading por estilo (sobre RGB, no sobre alpha)
        comp_rgb = grading(comp.convert("RGB"),
                           contrast=PRESETS[style]["contrast"],
                           saturation=PRESETS[style]["saturation"],
                           tone=PRESETS[style]["tone"])
        comp = comp_rgb.convert("RGBA")

        # 7) Export
        out_id = str(uuid4())
        out_path = os.path.join(OUTPUT_DIR, f"{out_id}.png")
        comp.save(out_path, "PNG", optimize=True)

        return {"status": "done", "url": f"/outputs/{out_id}.png", "width": w, "height": h}

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

# Health
@app.get("/")
def root():
    return {"ok": True, "endpoint": "/studio/context2d", "styles": list(PRESETS.keys())}
