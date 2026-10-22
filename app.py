import base64
import colorsys
import math
import random
import re
import smtplib
import sqlite3
import tempfile
import time
import urllib.parse
from datetime import datetime, time as dtime, timedelta, timezone
from email.message import EmailMessage
from io import BytesIO
from pathlib import Path

import pandas as pd
import requests
import streamlit as st
from PIL import Image, ImageOps
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from cake_render import render_cake_svg

st.set_page_config(page_title="AI Cake Assistant", page_icon="🎂", layout="centered")

# ---------------------------------------------------------------
# 1. CAKE DATASET  (35 flavours: price per kg + tags used by ML)
# ---------------------------------------------------------------
FLAVOURS = {
    # ---- your 20 flavours ----
    "Chocolate": (700, "chocolate rich classic birthday kids party"),
    "Choco Truffle": (850, "chocolate truffle ganache rich birthday anniversary"),
    "Double Chocolate": (900, "chocolate double rich birthday celebration"),
    "Lotus Biscoff": (1100, "biscoff caramel premium trendy birthday anniversary"),
    "Strawberry": (750, "strawberry fruit fresh cream birthday kids"),
    "Mango": (800, "mango fruit fresh summer birthday"),
    "Orange": (750, "orange fruit citrus fresh light"),
    "Nutella": (1100, "nutella hazelnut chocolate premium birthday anniversary"),
    "Cheesecake": (1200, "cheesecake creamy premium anniversary wedding"),
    "Baskin Burn Cheesecake": (1400, "cheesecake burnt premium creamy anniversary wedding"),
    "Vancho": (700, "vanilla chocolate mix classic kids birthday"),
    "Black Currant": (750, "blackcurrant fruit tangy birthday"),
    "Black Forest": (750, "black forest cherry chocolate classic birthday"),
    "White Forest": (750, "white forest cherry white chocolate classic birthday"),
    "Blueberry": (800, "blueberry fruit fresh light anniversary"),
    "Blue Velvet": (900, "blue velvet cream cheese colourful baby shower"),
    "Red Velvet": (950, "red velvet cream cheese romantic anniversary valentine wedding"),
    "Vanilla": (600, "vanilla classic simple light birthday kids wedding"),
    "Butterscotch": (700, "butterscotch caramel crunchy classic birthday"),
    "Rasamalai": (1000, "rasamalai indian traditional festival cardamom wedding"),
    # ---- 15 extra flavours ----
    "Dark Chocolate": (900, "dark chocolate bitter rich anniversary adults"),
    "White Chocolate": (900, "white chocolate sweet creamy wedding anniversary"),
    "Pineapple": (650, "pineapple fruit fresh classic birthday kids"),
    "Pista": (900, "pistachio nutty festival wedding premium"),
    "Butter Pecan": (950, "butter pecan nutty caramel anniversary"),
    "Coffee Mocha": (800, "coffee mocha chocolate adults farewell"),
    "Tiramisu": (1200, "tiramisu coffee italian premium anniversary"),
    "Caramel": (750, "caramel sweet classic birthday anniversary"),
    "Oreo": (850, "oreo cookies cream chocolate kids birthday"),
    "KitKat": (950, "kitkat chocolate wafer kids birthday party"),
    "Ferrero Rocher": (1300, "ferrero rocher hazelnut chocolate premium anniversary valentine"),
    "Rainbow": (850, "rainbow colourful kids birthday baby shower party"),
    "Blueberry Cheesecake": (1300, "blueberry cheesecake fruit creamy anniversary"),
    "Strawberry Cheesecake": (1300, "strawberry cheesecake fruit creamy valentine"),
    "Plum Fruit Cake": (900, "plum fruit cake christmas festival traditional wedding"),
}

# 9 cake types: extra charge (Rs)
CAKE_TYPES = {
    "Cheesecake": 150,
    "Regular Sponge Cake": 0,
    "Cupcakes": 0,
    "Jar Cake": 0,
    "Wedding Cake": 1500,
    "Anniversary Cake": 150,
    "Birthday Cake": 0,
    "Theme Cake": 500,
    "Drip Cake": 250,
}
OCCASIONS = ["Birthday", "Anniversary", "Wedding", "Baby Shower", "Valentine",
             "Festival", "Farewell", "Other"]
SHAPES = {"Round": 0, "Square": 0, "Rectangle": 50, "Heart": 100, "2 Tier": 600, "3 Tier": 1200}
FROSTINGS = ["Whipped Cream", "Butter Cream", "Chocolate Ganache", "Cream Cheese", "Fondant"]
TOPPINGS = {"Fresh Fruits": 100, "Chocolate Chips": 80, "Sprinkles": 50, "Oreo": 120,
            "Kitkat / Gems": 150, "Edible Flowers": 200, "Cherries": 60, "Nuts": 100,
            "Macarons": 180, "Pearls": 80, "Gold Foil": 150, "Rose Petals": 100,
            "Marshmallows": 80, "Biscuit Crumbs": 90}

INSTAGRAM_USER = "_femmora.bakes__"
INSTAGRAM_URL = f"https://www.instagram.com/{INSTAGRAM_USER}/"

PHOTO_DIR = Path(__file__).parent / "cake_photos"
PHOTO_EXT = {".jpg", ".jpeg", ".png", ".webp"}


# ---------------------------------------------------------------
# 2. HELPERS (data processing, ML, price, photo matching)
# ---------------------------------------------------------------
@st.cache_data
def flavour_table():
    return pd.DataFrame(
        [(n, v[0], v[1]) for n, v in FLAVOURS.items()],
        columns=["flavour", "price_per_kg", "tags"],
    )


def suggest_weight(people):
    return max(0.5, math.ceil(people / 10 * 2) / 2)  # ~10 people per kg


def estimate_price(per_kg, weight, cake_type, shape, toppings, topper):
    parts = [
        (f"Cake ({weight} kg x Rs.{per_kg})", per_kg * weight),
        (f"Cake type: {cake_type}", CAKE_TYPES[cake_type]),
        (f"Shape: {shape}", SHAPES[shape]),
    ]
    for t in toppings:
        parts.append((f"Topping: {t}", TOPPINGS[t]))
    if topper.strip():
        parts.append(("Cake topper / text", 100))
    total = sum(v for _, v in parts)
    return int(round(total / 5) * 5), parts


def recommend(flavour, occasion, cake_type, theme, top_n=3):
    df = flavour_table()
    docs = df["flavour"].str.lower() + " " + df["tags"]
    vec = TfidfVectorizer()
    matrix = vec.fit_transform(docs)
    query = vec.transform([f"{occasion} {flavour} {cake_type} {theme}".lower()])
    out = df.copy()
    out["score"] = cosine_similarity(query, matrix).flatten()
    out = out[out["flavour"] != flavour]
    return out.sort_values("score", ascending=False).head(top_n)


def compact(text):
    """'Black Forest' / 'black-forest' / 'black_forest' -> 'blackforest'."""
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def _longest_match(name, keys):
    hits = [k for k in keys if k and k in name]
    return max(hits, key=len) if hits else None


def find_photos(flavour, cake_type, shape, limit=6):
    """Match photos in cake_photos/ by file name. No hyphen needed:
    blackforest1.jpg, Black Forest 2.jpg, heart_redvelvet.jpg all work."""
    if not PHOTO_DIR.exists():
        return []
    flav_keys = [compact(f) for f in FLAVOURS]
    type_keys = [compact(t) for t in CAKE_TYPES] + [compact(t).replace("cake", "") for t in CAKE_TYPES]
    want_flav = compact(flavour)
    want_types = {compact(cake_type), compact(cake_type).replace("cake", "")}
    want_shape = compact(shape)
    scored = []
    for p in PHOTO_DIR.iterdir():
        if p.suffix.lower() not in PHOTO_EXT:
            continue
        n = compact(p.stem)
        photo_flav = _longest_match(n, flav_keys)
        if photo_flav and photo_flav != want_flav:
            continue  # photo belongs to another flavour
        photo_type = _longest_match(n, type_keys)
        score = 0
        if photo_flav == want_flav:
            score += 4
        if photo_type and photo_type in want_types:
            score += 2
        if want_shape in n:
            score += 1
        if score:
            scored.append((score, p))
    scored.sort(key=lambda x: (-x[0], x[1].name))
    return [p for _, p in scored[:limit]]


def photo_checklist_csv():
    rows = ["category,name,example_file_name"]
    for f in FLAVOURS:
        rows.append(f"flavour,{f},{compact(f)}1.jpg")
    for t in CAKE_TYPES:
        rows.append(f"type,{t},{compact(t)}1.jpg")
    for sh in SHAPES:
        rows.append(f"shape,{sh},{compact(sh)}1.jpg")
    return "\n".join(rows)


def load_reference(file):
    """Open + shrink the uploaded photo so big phone pictures don't hang."""
    try:
        img = Image.open(file)
        img = ImageOps.exif_transpose(img)
        img.thumbnail((800, 800))
        return img.convert("RGB")
    except Exception:
        return None


def nearest_colour_name(hex_colour):
    """Hex -> closest named colour (so the AI prompt says 'teal', not '#2a9d8f')."""
    h = hex_colour.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    best, best_d = "pink", 10 ** 9
    for n, v in QUICK.items():
        if not v:
            continue
        vh = v.lstrip("#")
        vr, vg, vb = int(vh[0:2], 16), int(vh[2:4], 16), int(vh[4:6], 16)
        d = (r - vr) ** 2 + (g - vg) ** 2 + (b - vb) ** 2
        if d < best_d:
            best, best_d = n, d
    return best.lower()


def prompt_colour_name(hex_colour):
    """Hex -> clear colour words for the AI prompt (e.g. #dc1858 -> 'vivid crimson red')."""
    h = hex_colour.lstrip("#")
    r, g, b = int(h[0:2], 16) / 255, int(h[2:4], 16) / 255, int(h[4:6], 16) / 255
    hh, sat, val = colorsys.rgb_to_hsv(r, g, b)
    hue = hh * 360
    if val < 0.18:
        return "black"
    if sat < 0.10:
        return "white" if val > 0.9 else ("light grey" if val > 0.7 else ("grey" if val > 0.4 else "charcoal"))
    if 15 <= hue < 45 and val < 0.65:
        base = "brown"
    else:
        base = "red"
        for bound, name in [(15, "red"), (40, "orange"), (65, "yellow"), (90, "lime green"), (160, "green"),
                            (195, "teal"), (250, "blue"), (285, "purple"), (325, "magenta pink"),
                            (345, "crimson red"), (361, "red")]:
            if hue < bound:
                base = name
                break
    if base in ("magenta pink", "crimson red") and sat < 0.6:
        base = "pink"
    if val < 0.5:
        q = "dark "
    elif sat < 0.45:
        q = "pastel "
    elif sat > 0.75 and val > 0.75:
        q = "vivid "
    else:
        q = ""
    return q + base


# ---------------------------------------------------------------
# AI IMAGE: accurate prompt + reliable fetch
# ---------------------------------------------------------------
SHAPE_WORDS = {"Round": "round", "Square": "square", "Rectangle": "rectangular", "Heart": "heart-shaped",
               "2 Tier": "two-tier", "3 Tier": "three-tier"}

# what each cake type should look like
TYPE_LOOK = {
    "Cheesecake": "a {shape} {cheese} with a smooth creamy top and a biscuit crust base",
    "Regular Sponge Cake": "a {shape} sponge cake with smooth straight sides and a flat top",
    "Cupcakes": "six cupcakes in paper liners, each with a tall frosting swirl piped with a star nozzle",
    "Jar Cake": "three dessert cakes in clear glass jars with visible cake and cream layers",
    "Wedding Cake": "an elegant {shape} wedding cake with smooth tiers and clean edges",
    "Anniversary Cake": "a {shape} anniversary cake with smooth frosting and elegant decoration",
    "Birthday Cake": "a {shape} birthday cake with smooth frosting",
    "Theme Cake": "a {shape} cake with neat detailed decoration and clean edges",
    "Drip Cake": "a tall {shape} drip cake with smooth sides and glossy ganache dripping evenly from the top edge",
}

# simple words so the AI model draws each topping correctly
TOPPING_WORDS = {
    "Fresh Fruits": "fresh strawberries and berries", "Chocolate Chips": "chocolate chips",
    "Sprinkles": "colourful sprinkles", "Oreo": "oreo cookies", "Kitkat / Gems": "kitkat fingers and candy gems",
    "Edible Flowers": "small edible flowers", "Cherries": "red cherries", "Nuts": "chopped nuts",
    "Macarons": "pastel macarons", "Pearls": "white sugar pearls", "Gold Foil": "gold leaf flakes",
    "Rose Petals": "rose petals", "Marshmallows": "mini marshmallows", "Biscuit Crumbs": "biscuit crumbs",
}

# how the frosting / piping should look (neat, like a real bakery cake)
PIPING_LOOK = {
    "Whipped Cream": "soft whipped cream, smooth sides, neat piped shell border at top and base, small sharp star-tip rosettes",
    "Butter Cream": "smooth buttercream, crisp sharply defined piped rosettes, neat shell border at top edge and base, clean star-tip piping, perfectly even",
    "Chocolate Ganache": "glossy smooth chocolate ganache, neat piped ganache swirls, mirror-like even finish",
    "Cream Cheese": "thick smooth cream cheese frosting, clean swirled top, neat evenly spaced piped dollops",
    "Fondant": "smooth matte fondant covering with sharp clean edges and neat piped royal icing borders",
}

NEGATIVE_PROMPT = ("blurry, distorted, deformed, melted, messy frosting, uneven piping, sloppy piping, wrong colour, "
                   "extra layers, cartoon, illustration, painting, 3d render, text, letters, watermark, logo, "
                   "people, hands, duplicate cake")


def size_word(weight):
    if weight <= 0.5:
        return "small compact (about 6 inch)"
    if weight <= 1:
        return "medium (about 7 inch)"
    if weight <= 2:
        return "large (about 9 inch)"
    return "extra large (about 11 inch)"


def realistic_prompt(colour_name, colour_hex, shape, flavour, cake_type, frosting, toppings, theme, occasion, weight):
    """Short, clear prompt: image models follow short prompts much better than long ones."""
    fl = flavour.lower()
    cheese = fl if "cheesecake" in fl else f"{fl} cheesecake"
    what = TYPE_LOOK.get(cake_type, "a {shape} cake").format(shape=SHAPE_WORDS[shape], cheese=cheese)
    parts = [
        f"{what}, {colour_name} coloured, professional bakery photo",
        f"the entire cake is {colour_name} coloured, {colour_name} {frosting.lower()} on every surface",
        PIPING_LOOK.get(frosting, frosting.lower()),
        f"{size_word(weight)} cake",
    ]
    if toppings:
        parts.append("decorated with only " + ", ".join(TOPPING_WORDS.get(t, t.lower()) for t in toppings))
    else:
        parts.append("no extra decorations")
    if theme.strip():
        parts.append(f"{theme.strip()} style")
    parts.append(f"the cake is clearly {colour_name}, plain white cake stand, plain soft neutral background, one single cake centered, sharp focus, "
                 "natural light, real bakery photograph, no text")
    return ", ".join(parts)


def realistic_url(prompt, seed, model="flux", size=1024):
    return ("https://image.pollinations.ai/prompt/" + urllib.parse.quote(prompt)
            + f"?width={size}&height={size}&nologo=true&private=true&model={model}&seed={seed}")


def hf_image(prompt, seed):
    """Optional better quality: free Hugging Face token (HF_TOKEN in Streamlit secrets)."""
    token = get_secret("HF_TOKEN")
    if not token:
        return None, ""
    try:
        r = requests.post(
            "https://router.huggingface.co/hf-inference/models/black-forest-labs/FLUX.1-schnell",
            headers={"Authorization": f"Bearer {token}", "Accept": "image/png"},
            json={"inputs": prompt, "parameters": {"seed": seed}}, timeout=60)
        if r.status_code == 200 and r.headers.get("content-type", "").startswith("image"):
            return r.content, ""
        return None, f"HF: HTTP {r.status_code}"
    except Exception as e:
        return None, f"HF: {e}"


def generate_realistic_image(prompt, seed, model):
    """Fetch the image on the server (retry + fallback) so every click gives a fresh image."""
    img, err = hf_image(prompt, seed)
    if img:
        return img, "", ""
    last_err = err
    order = [model] + [m for m in ("flux", "turbo") if m != model]
    for m in order:
        url = realistic_url(prompt, seed, m, 1024 if m == "flux" else 768)
        for attempt in range(2):
            try:
                r = requests.get(url, timeout=60)
                if r.status_code == 200 and r.headers.get("content-type", "").startswith("image"):
                    return r.content, url, ""
                last_err = f"{m}: HTTP {r.status_code}"
                if r.status_code == 429:
                    time.sleep(4)  # rate limited, wait and retry once
                    continue
                break
            except Exception as e:
                last_err = f"{m}: {e}"
                break
    return None, "", last_err


IST = timezone(timedelta(hours=5, minutes=30))


def get_secret(key):
    try:
        return st.secrets[key]
    except Exception:
        return None


def notify_owner(text, photo_bytes=None, svg_text=None, order_id=""):
    """Send the order to the owner's Telegram. Returns (ok, error_message)."""
    token = get_secret("TELEGRAM_BOT_TOKEN")
    chat = get_secret("TELEGRAM_CHAT_ID")
    if not token or not chat:
        return False, "Owner Telegram is not set up yet."
    base = f"https://api.telegram.org/bot{token}"
    try:
        r = requests.post(f"{base}/sendMessage", data={"chat_id": chat, "text": text}, timeout=20)
        if not (r.ok and r.json().get("ok")):
            return False, f"Telegram error: {r.text[:150]}"
        if photo_bytes:
            requests.post(f"{base}/sendPhoto", data={"chat_id": chat, "caption": f"Reference image - {order_id}"},
                          files={"photo": ("reference.jpg", photo_bytes)}, timeout=40)
        if svg_text:
            requests.post(f"{base}/sendDocument", data={"chat_id": chat, "caption": f"Cake design - {order_id} (open in browser)"},
                          files={"document": (f"{order_id}_design.svg", svg_text.encode(), "image/svg+xml")}, timeout=40)
        return True, ""
    except Exception as e:
        return False, f"Could not reach Telegram: {e}"


STATUSES = ["New", "Confirmed", "Preparing", "Ready", "Delivered", "Cancelled"]


def _db_path():
    p = Path(__file__).parent / "orders.db"
    try:
        with open(p, "ab"):
            pass
        return str(p)
    except Exception:
        return str(Path(tempfile.gettempdir()) / "orders.db")


def _db():
    con = sqlite3.connect(_db_path())
    con.execute(
        "CREATE TABLE IF NOT EXISTS orders (order_id TEXT PRIMARY KEY, created TEXT, name TEXT, phone TEXT, "
        "delivery TEXT, need TEXT, cake_type TEXT, flavour TEXT, weight REAL, price INTEGER, status TEXT, "
        "details TEXT, photo BLOB, design TEXT)"
    )
    return con


def save_order(o):
    try:
        con = _db()
        con.execute(
            "INSERT INTO orders VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (o["order_id"], o["created"], o["name"], o["phone"], o["delivery"], o["need"], o["cake_type"],
             o["flavour"], o["weight"], o["price"], "New", o["details"], o["photo"], o["design"]),
        )
        con.commit()
        con.close()
        return True, ""
    except Exception as e:
        return False, str(e)


def list_orders():
    con = _db()
    df = pd.read_sql_query(
        "SELECT order_id, created, name, phone, delivery, need, cake_type, flavour, weight, price, status "
        "FROM orders ORDER BY created DESC", con)
    con.close()
    return df


def get_order(order_id):
    con = _db()
    con.row_factory = sqlite3.Row
    row = con.execute("SELECT * FROM orders WHERE order_id=?", (order_id,)).fetchone()
    con.close()
    return dict(row) if row else None


def update_status(order_id, status):
    con = _db()
    con.execute("UPDATE orders SET status=? WHERE order_id=?", (status, order_id))
    con.commit()
    con.close()


def email_owner(subject, text, photo_bytes=None, svg_text=None, order_id=""):
    """Email the order to the owner (Gmail + App Password)."""
    user = get_secret("EMAIL_USER")
    pw = get_secret("EMAIL_APP_PASSWORD")
    to = get_secret("OWNER_EMAIL") or user
    if not user or not pw:
        return False, "Email is not set up yet."
    try:
        msg = EmailMessage()
        msg["Subject"], msg["From"], msg["To"] = subject, user, to
        msg.set_content(text)
        if photo_bytes:
            msg.add_attachment(photo_bytes, maintype="image", subtype="jpeg", filename="reference.jpg")
        if svg_text:
            msg.add_attachment(svg_text.encode(), maintype="image", subtype="svg+xml",
                               filename=f"{order_id}_design.svg")
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=20) as server:
            server.login(user, str(pw).replace(" ", ""))
            server.send_message(msg)
        return True, ""
    except Exception as e:
        return False, f"Email error: {e}"


def owner_dashboard():
    st.title("🔐 Owner Dashboard")
    pw = get_secret("OWNER_PASSWORD")
    if not pw:
        st.error("Owner password is not set. Add OWNER_PASSWORD in Streamlit Secrets.")
        return
    if not st.session_state.get("owner_ok"):
        entered = st.text_input("Owner password", type="password")
        if st.button("Login"):
            if entered == str(pw):
                st.session_state["owner_ok"] = True
                st.rerun()
            else:
                st.error("Wrong password")
        return
    if st.button("Logout"):
        st.session_state["owner_ok"] = False
        st.rerun()

    df = list_orders()
    if df.empty:
        st.info("No orders yet.")
        return
    c1, c2, c3 = st.columns(3)
    c1.metric("Total orders", len(df))
    c2.metric("New", int((df["status"] == "New").sum()))
    c3.metric("Est. revenue", f"₹{int(df['price'].sum())}")
    flt = st.selectbox("Filter by status", ["All"] + STATUSES)
    view = df if flt == "All" else df[df["status"] == flt]
    st.dataframe(view, use_container_width=True, hide_index=True)
    st.download_button("⬇️ Download all orders (CSV)", df.to_csv(index=False), "orders.csv", "text/csv")
    st.caption("Note: orders are kept while the app is running. Download the CSV regularly; your email keeps a permanent copy.")
    if view.empty:
        return

    st.subheader("Order details")
    oid = st.selectbox("Select order", view["order_id"].tolist())
    o = get_order(oid)
    st.text(o["details"])
    if o["photo"]:
        st.image(o["photo"], caption="Customer reference image", width=300)
    if o["design"]:
        st.image(o["design"], caption="Cake design preview", use_container_width=True)
    new_status = st.selectbox("Order status", STATUSES, index=STATUSES.index(o["status"]), key=f"st_{oid}")
    if st.button("Update status"):
        update_status(oid, new_status)
        st.success(f"{oid} -> {new_status}")
    digits = re.sub(r"\D", "", o["phone"])
    if len(digits) == 10:
        digits = "91" + digits
    st.link_button("💬 Message customer on WhatsApp", f"https://wa.me/{digits}")


def show_svg(svg):
    st.image(svg, use_container_width=True)


# ---------------------------------------------------------------
# 3. USER INPUT
# ---------------------------------------------------------------
menu = st.sidebar.radio("Menu", ["🎂 Customer", "🔐 Owner Dashboard"])
if menu == "🔐 Owner Dashboard":
    owner_dashboard()
    st.stop()

st.title("🎂 AI Bakery Recommendation & Cake Design Assistant")
st.caption("Customise your cake - the preview updates as you choose.")
st.link_button(f"📸 Follow @{INSTAGRAM_USER} on Instagram", INSTAGRAM_URL)

st.subheader("👤 Customer Details")
name = st.text_input("Name")
phone = st.text_input("Phone Number")

st.subheader("🍰 Cake Details")
cake_type = st.selectbox("Cake Type", list(CAKE_TYPES.keys()))
flavour = st.selectbox("Cake Flavour (35 flavours)", list(FLAVOURS.keys()))
people = st.number_input("Number of People", min_value=1, max_value=1000, value=10)
suggested = suggest_weight(people)
SIZES = {"Auto (based on people)": None, "½ kg (500 g)": 0.5, "1 kg": 1.0, "1½ kg": 1.5,
         "2 kg": 2.0, "3 kg": 3.0, "4 kg": 4.0, "5 kg": 5.0, "Custom weight": "custom"}
size_choice = st.selectbox("Cake Size / Weight", list(SIZES.keys()))
if SIZES[size_choice] == "custom":
    weight = st.number_input("Enter any weight (kg)", min_value=0.25, max_value=50.0,
                             value=float(suggested), step=0.25)
elif SIZES[size_choice] is None:
    weight = float(suggested)
else:
    weight = float(SIZES[size_choice])
st.caption(f"💡 For {people} people, about {suggested} kg is suggested. Selected: {weight} kg")
occasion = st.selectbox("Occasion", OCCASIONS)
budget = st.number_input("Budget (₹)", min_value=300, max_value=100000, value=1500, step=100)

st.subheader("🎨 Design")
shape = st.selectbox("Cake Shape", list(SHAPES.keys()))
QUICK = {
    "Custom (use picker below)": None, "Pink": "#f48fb1", "Baby Pink": "#f8c8dc", "Hot Pink": "#ff3d8b",
    "Red": "#d62839", "Maroon": "#7b1e2b", "Orange": "#ff9f45", "Peach": "#ffcba4",
    "Yellow": "#ffd166", "Gold": "#d4af37", "Cream": "#f5e6c8", "White": "#fffaf0",
    "Mint Green": "#a8e6cf", "Green": "#7bc67b", "Dark Green": "#2d6a4f", "Teal": "#2a9d8f",
    "Sky Blue": "#8ecae6", "Blue": "#3a86ff", "Navy": "#1d3557", "Lavender": "#c8b6ff",
    "Purple": "#9b5de5", "Violet": "#6a4c93", "Brown": "#6b3e26", "Chocolate": "#3e2415",
    "Caramel": "#c68b59", "Grey": "#adb5bd", "Black": "#222222", "Rose Gold": "#e0a899",
    "Silver": "#c0c0c0",
}
quick = st.selectbox("Cake Colour (28 colours)", list(QUICK.keys()))
picked = st.color_picker("Custom colour picker (used when 'Custom' is selected)", "#f48fb1")
colour = QUICK[quick] or picked
st.markdown(
    f'<div style="height:30px;border-radius:8px;border:1px solid #888;background:{colour}"></div>',
    unsafe_allow_html=True,
)
st.caption(f"Selected colour: {quick if QUICK[quick] else 'Custom'} ({colour})")
theme = st.text_input("Design / Theme (e.g. Floral, Heart, Star, Frozen)")
frosting = st.selectbox("Frosting / Cream", FROSTINGS)
toppings = st.multiselect("Toppings / Decorations", list(TOPPINGS.keys()))
topper = st.text_input("Cake Topper / Text (shown on top, e.g. Happy Birthday Riya)")
message = st.text_input("Customisation Message (written on cake)")
ref_file = st.file_uploader("Reference Cake Image (optional)")

# ---------------------------------------------------------------
# 4. OUTPUT
# ---------------------------------------------------------------
st.divider()
st.header("📸 Matching Cake Photos")
photos = find_photos(flavour, cake_type, shape)
if photos:
    cols = st.columns(2)
    for i, p in enumerate(photos):
        cols[i % 2].image(str(p), use_container_width=True)
else:
    st.info("No matching photos yet. Add photos to the 'cake_photos' folder in GitHub "
            "(file name like blackforest1.jpg or heart1.jpg - no hyphen needed).")

st.header("🖼️ Colour Preview (your colour & design)")
svg = render_cake_svg(
    colour=colour, shape=shape, cake_type=cake_type, weight=weight,
    toppings=toppings, topper=topper, message=message, occasion=occasion, theme=theme,
)
show_svg(svg)
st.caption(f"{colour} {shape} {flavour} {cake_type} • {weight} kg • {frosting}")

st.header("✨ Realistic Cake Image (AI)")
st.caption("Free AI image - no API key needed. It follows your cake type, size, colour, shape, frosting and toppings. "
           "Text on the cake is shown in the drawing above because AI cannot spell correctly.")
ai_model = st.selectbox("AI image model", ["flux", "turbo"],
                        help="'flux' = more accurate. 'turbo' = faster. If one fails, the other is tried automatically.")
if st.button("✨ Generate / Regenerate realistic image"):
    cname = quick.lower() if QUICK[quick] else prompt_colour_name(colour)
    p = realistic_prompt(cname, colour, shape, flavour, cake_type, frosting, toppings, theme, occasion, weight)
    seed = random.randint(1, 999999)
    with st.spinner("Generating your cake image... (10-30 seconds)"):
        img_bytes, used_url, err = generate_realistic_image(p, seed, ai_model)
    if img_bytes:
        st.session_state["ai_img"] = img_bytes
        st.session_state["ai_url"] = used_url
        st.session_state["ai_prompt"] = p
    else:
        st.error("Image could not be generated right now. Please press the button again in a few seconds.")
        st.caption(err)
if "ai_img" in st.session_state:
    st.image(st.session_state["ai_img"], use_container_width=True,
             caption="AI generated. Press the button again for a fresh version.")
    with st.expander("Prompt used"):
        st.write(st.session_state["ai_prompt"])
    st.caption("If you change type/colour/shape/toppings, press the button again to update this image.")

price, parts = estimate_price(FLAVOURS[flavour][0], weight, cake_type, shape, toppings, topper)

st.header("📋 Final Recommendation")
if name:
    st.write(f"Hello **{name}**, here is your cake summary:")
st.write(f"**Cake Type:** {cake_type}")
st.write(f"**Flavour:** {flavour}")
st.write(f"**Size / Weight:** {weight} kg (for about {people} people)")
st.write(f"**Shape:** {shape}  |  **Frosting:** {frosting}")
if theme:
    st.write(f"**Theme:** {theme}")
if toppings:
    st.write(f"**Toppings:** {', '.join(toppings)}")
if topper:
    st.write(f"**Topper Text:** {topper}")
if message:
    st.write(f"**Message:** {message}")
st.success(f"💰 Estimated Price: ₹{price}")
if price > budget:
    st.warning(f"This is ₹{price - budget} above your budget. Try fewer toppings, a simpler shape or less weight.")
with st.expander("Price details"):
    for label, value in parts:
        st.write(f"{label}: ₹{int(value)}")

st.subheader("⭐ You may also like")
for _, r in recommend(flavour, occasion, cake_type, theme).iterrows():
    alt_price, _ = estimate_price(r["price_per_kg"], weight, cake_type, shape, toppings, topper)
    st.write(f"• **{r['flavour']}** - ₹{alt_price}")

if ref_file is not None:
    st.subheader("📷 Your Reference Image")
    ref = load_reference(ref_file)
    if ref is None:
        st.error("This file could not be read as an image. Please upload a JPG or PNG photo.")
    else:
        st.image(ref, width=300)

st.divider()
st.header("🛒 Place Order")
st.caption("Check your cake above, fill these details and press Place Order. The bakery gets your order instantly.")
delivery_type = st.radio("Pickup or Delivery", ["Pickup from bakery", "Home delivery"], horizontal=True)
address = st.text_area("Delivery address") if delivery_type == "Home delivery" else ""
today = datetime.now(IST).date()
need_date = st.date_input("Required date", value=today + timedelta(days=1), min_value=today)
need_time = st.time_input("Required time", value=dtime(10, 0))
notes = st.text_input("Extra note for the baker (optional)")

if st.button("✅ Place Order", type="primary"):
    digits = re.sub(r"\D", "", phone)
    if not name.strip():
        st.error("Please enter your Name (in Customer Details at the top).")
    elif not (10 <= len(digits) <= 13):
        st.error("Please enter a valid Phone Number (in Customer Details at the top).")
    elif delivery_type == "Home delivery" and not address.strip():
        st.error("Please enter the delivery address.")
    else:
        order_id = "CK" + datetime.now(IST).strftime("%d%m%H%M%S")
        colour_label = quick if QUICK[quick] else nearest_colour_name(colour)
        lines = [
            f"🎂 NEW CAKE ORDER  #{order_id}",
            f"Customer: {name.strip()}",
            f"Phone: {phone.strip()}",
            f"{delivery_type}" + (f": {address.strip()}" if address.strip() else ""),
            f"Required: {need_date.strftime('%d %b %Y')} at {need_time.strftime('%I:%M %p')}",
            "",
            f"Cake Type: {cake_type}",
            f"Flavour: {flavour}",
            f"Weight: {weight} kg (for ~{people} people)",
            f"Shape: {shape}",
            f"Colour: {colour_label} ({colour})",
            f"Frosting: {frosting}",
            f"Occasion: {occasion}",
            f"Theme: {theme or '-'}",
            f"Toppings: {', '.join(toppings) if toppings else '-'}",
            f"Topper text: {topper or '-'}",
            f"Message on cake: {message or '-'}",
            f"Notes: {notes or '-'}",
            "",
            f"Estimated Price: Rs.{price}  (budget: Rs.{budget})",
        ]
        if st.session_state.get("ai_url"):
            lines += ["", f"AI preview: {st.session_state['ai_url']}"]
        text = "\n".join(lines)

        photo_bytes = None
        if ref_file is not None:
            ref_img = load_reference(ref_file)
            if ref_img is not None:
                buf = BytesIO()
                ref_img.save(buf, format="JPEG", quality=85)
                photo_bytes = buf.getvalue()

        with st.spinner("Placing your order..."):
            saved, db_err = save_order({
                "order_id": order_id, "created": datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S"),
                "name": name.strip(), "phone": phone.strip(), "delivery": delivery_type,
                "need": f"{need_date.strftime('%d %b %Y')} {need_time.strftime('%I:%M %p')}",
                "cake_type": cake_type, "flavour": flavour, "weight": float(weight), "price": int(price),
                "details": text, "photo": photo_bytes, "design": svg,
            })
            subject = f"New Cake Order #{order_id} - {name.strip()} - Rs.{price}"
            email_owner(subject, text, photo_bytes, svg, order_id)
            notify_owner(text, photo_bytes, svg, order_id)
        if saved:
            st.success(f"🎉 Order placed! Your Order ID is **{order_id}**. The bakery will contact you on {phone.strip()}.")
            st.balloons()
            st.link_button(f"📸 Follow @{INSTAGRAM_USER} on Instagram", INSTAGRAM_URL, key="ig_after_order")
        else:
            st.error("Sorry, we could not save your order. Please send it on WhatsApp using the button below.")
            st.caption(db_err)
        owner_wa = get_secret("OWNER_WHATSAPP")
        if owner_wa:
            st.link_button("📲 Also send order on WhatsApp",
                           f"https://wa.me/{owner_wa}?text=" + urllib.parse.quote(text))
        with st.expander("Your order summary"):
            st.text(text)

st.divider()
st.markdown(f"🎂 **Femmora Bakes** • [Instagram @{INSTAGRAM_USER}]({INSTAGRAM_URL})")

with st.expander("📝 Photo names checklist (35 flavours, 9 types, shapes)"):
    st.download_button("Download checklist (CSV)", photo_checklist_csv(), "photo_checklist.csv", "text/csv")
