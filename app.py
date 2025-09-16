import os, re, json
from flask import Flask, render_template, request, jsonify

# حمّل متغيّرات البيئة من .env محليًا
from dotenv import load_dotenv
load_dotenv()

# ====== NEW: استيراد موصل MySQL ======
try:
    import mysql.connector
except Exception:
    mysql = None
# =====================================

app = Flask(__name__)

# ========= المفتاح من ENV وليس من الكود =========
API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or os.getenv("API_KEY")
if not API_KEY:
    raise RuntimeError("فضلاً ضَع مفتاح Gemini في متغيّر البيئة GEMINI_API_KEY (أو GOOGLE_API_KEY).")
# ===============================================

MODEL_NAME = os.getenv("MODEL_NAME", "gemini-2.5-flash")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 3000))  # كان 6000
OVERLAP    = int(os.getenv("OVERLAP", 32))       # كان 80
AR_WORD    = re.compile(r'[\u0621-\u064A]+', re.UNICODE)

# أقواس JSON مُضاعفة حتى لا تكسرها str.format
PROMPT_TMPL = """أنت مدقق إملائي/نحوي عربي وإنجليزي.
أعد استجابة بصيغة JSON نقي فقط (بدون أي تعليقات أو أسطر إضافية أو ```).

{{
  "matches": [
    {{
      "offset": 0,
      "length": 0,
      "surface": "",
      "message": "",
      "replacements": [],
      "rule_id": "GEMINI_25_FLASH_AR"
    }}
  ]
}}

المطلوب:
- أعد كائن JSON واحد فيه مصفوفة "matches".
- إن لم توجد أخطاء: {{"matches":[]}}
- لا تُدرج نصًا مصححًا كاملًا ولا تضع ```json.

النص:
<<<
{snippet}
>>>
"""



def build_prompt(snippet: str) -> str:
    return PROMPT_TMPL.format(snippet=snippet)

def chunk_text_with_overlap(text: str, size: int = CHUNK_SIZE, overlap: int = OVERLAP):
    """يرجع [(base_offset, chunk_text)]. آخر كل قطعة يتداخل مع بداية التالية."""
    chunks, start, n = [], 0, len(text)
    while start < n:
        end = min(start + size, n)
        # حاول القطع عند حدود المسافات
        if end < n and not text[end-1].isspace():
            back = text.rfind(" ", start, end)
            if back > start + size // 2:
                end = back + 1
        chunks.append((start, text[start:end]))
        if end == n:
            break
        start = max(end - overlap, start + 1)
    return chunks

def _extract_json(raw: str) -> dict:
    if not raw:
        return {}
    s = str(raw)

    # أزل حواجز الكود والتعليقات والفواصل الزائدة
    s = re.sub(r"^```(?:json)?\s*|\s*```$", "", s.strip(), flags=re.IGNORECASE|re.MULTILINE)
    s = re.sub(r"//.*?$", "", s, flags=re.MULTILINE)
    s = re.sub(r"/\*[\s\S]*?\*/", "", s)
    s = re.sub(r",\s*([}\]])", r"\1", s)

    # 1) حاول مباشرة
    try:
        return json.loads(s)
    except Exception:
        pass

    # 2) التقط matches[] ولفّها في كائن
    m = re.search(r'"matches"\s*:\s*(\[[\s\S]*?\])', s)
    if m:
        arr = m.group(1)
        try:
            return {"matches": json.loads(arr)}
        except Exception:
            # جرّب إصلاح مفاتيح غير مقتبسة وأقواس منفردة (تقريبية)
            fix = re.sub(r"(?<=\{|,)\s*([A-Za-z_]\w*)\s*:", r'"\1":', arr)  # اقتبس المفاتيح
            fix = re.sub(r"'", r'"', fix)                                  # استبدل ' بـ "
            fix = re.sub(r",\s*([}\]])", r"\1", fix)                       # شِيل الفواصل الأخيرة
            try:
                return {"matches": json.loads(fix)}
            except Exception:
                return {}

    # 3) التقط أول كائن {} إن وُجد
    m2 = re.search(r"\{[\s\S]*\}", s)
    if m2:
        t = m2.group(0)
        t = re.sub(r"//.*?$", "", t, flags=re.MULTILINE)
        t = re.sub(r"/\*[\s\S]*?\*/", "", t)
        t = re.sub(r",\s*([}\]])", r"\1", t)
        try:
            return json.loads(t)
        except Exception:
            return {}

    return {}





def gen_gemini_response(snippet: str) -> str:
    """
    يستدعي Gemini ويُرجع JSON مضبوط حسب مخطط (بدون تعليقات أو زوائد).
    """
    prompt = build_prompt(snippet)

    # المحاولة 1: google.generativeai مع response_schema
    try:
        import google.generativeai as genai
        from google.generativeai import types as ggtypes

        genai.configure(api_key=API_KEY)
        model_name = (os.getenv("MODEL_NAME") or "gemini-1.5-flash").strip()

        schema = ggtypes.Schema(  # مخطط الاستجابة
            type=ggtypes.Type.OBJECT,
            properties={
                "matches": ggtypes.Schema(
                    type=ggtypes.Type.ARRAY,
                    items=ggtypes.Schema(
                        type=ggtypes.Type.OBJECT,
                        properties={
                            "offset": ggtypes.Schema(type=ggtypes.Type.INTEGER),
                            "length": ggtypes.Schema(type=ggtypes.Type.INTEGER),
                            "surface": ggtypes.Schema(type=ggtypes.Type.STRING),
                            "message": ggtypes.Schema(type=ggtypes.Type.STRING),
                            "replacements": ggtypes.Schema(
                                type=ggtypes.Type.ARRAY,
                                items=ggtypes.Schema(type=ggtypes.Type.STRING),
                            ),
                            "rule_id": ggtypes.Schema(type=ggtypes.Type.STRING),
                        },
                        required=["offset","length","surface","message","replacements","rule_id"],
                    ),
                ),
            },
            required=["matches"],
        )

        cfg = genai.GenerationConfig(
            response_mime_type="application/json",
            response_schema=schema,
            temperature=0,
        )

        model = genai.GenerativeModel(model_name)
        resp = model.generate_content(prompt, generation_config=cfg)
        return (resp.text or "{}")
    except Exception as e:
        print("generativeai(schema) error:", e)

    # المحاولة 2: google.genai كبديل
    try:
        from google import genai
        from google.genai import types
        client = genai.Client(api_key=API_KEY)
        model_name = (os.getenv("MODEL_NAME") or "gemini-2.0-flash").strip()
        cfg = types.GenerateContentConfig(response_mime_type="application/json", temperature=0)
        resp = client.models.generate_content(model=model_name, contents=prompt, config=cfg)
        return resp.text or "{}"
    except Exception as e:
        print("genai(new) error:", e)
        return "{}"



def align_matches(snippet: str, raw_matches: list) -> list:
    """
    يثبت offsets بالبحث عن surface داخل المقطع إذا أعاد النموذج موضعًا خاطئًا.
    يمنع التداخل المزدوج داخل نفس المقطع.
    """
    def is_free(a, b, used):
        for u0, u1 in used:
            if not (b <= u0 or a >= u1):
                return False
        return True

    used = []
    aligned = []

    for m in raw_matches:
        try:
            # اقرأ المعطيات
            off = int(m.get("offset", -1))
            ln  = int(m.get("length", 0))
            surface = (m.get("surface") or "").strip()
            msg = str(m.get("message", "تصحيح محتمل"))
            reps = [str(r) for r in (m.get("replacements") or [])][:5]

            # إن لم يوجد surface، حاول اقتطاعه من offset/length (إن كانا صالحَين)
            if not surface and (0 <= off < len(snippet)) and ln > 0 and off + ln <= len(snippet):
                surface = snippet[off:off+ln]

            if not surface:
                continue

            # تحقّق الموضع
            good = (0 <= off < len(snippet)) and ln > 0 and (off + ln) <= len(snippet) and snippet[off:off+ln] == surface
            if not good:
                # ابحث عن أقرب ظهور لـ surface غير مستخدم
                pos = snippet.find(surface)
                found = False
                while pos != -1:
                    a, b = pos, pos + len(surface)
                    if is_free(a, b, used):
                        off, ln = a, len(surface)
                        found = True
                        break
                    pos = snippet.find(surface, pos + 1)
                if not found:
                    continue  # لا نستخدم مطابقة غير قابلة للمحاذاة

            # منع تداخل مزدوج
            if not is_free(off, off + ln, used):
                continue

            used.append((off, off + ln))
            aligned.append({
                "offset": off,
                "length": ln,
                "message": msg,
                "replacements": [r for r in reps if r and r != surface][:5],
                "rule_id": "GEMINI_25_FLASH_AR"
            })
        except Exception:
            continue

    # ترتيب نهائي
    aligned.sort(key=lambda x: x["offset"])
    return aligned

def gemini_check_chunk(snippet: str) -> list:
    raw = gen_gemini_response(snippet)
    data = _extract_json(raw)
    raw_matches = data.get("matches", []) if isinstance(data, dict) else []
    return align_matches(snippet, raw_matches)

# ========== NEW: إعداد اتصال MySQL من ENV ==========
DB_HOST  = os.getenv("DB_HOST", "127.0.0.1")
DB_USER  = os.getenv("DB_USER", "")
DB_PASS  = os.getenv("DB_PASSWORD", "")
DB_NAME  = os.getenv("DB_NAME", "")
DB_TABLE = os.getenv("DB_TABLE", "users")  # اسم الجدول الذي يحتوي المستخدمين

def get_db():
    """إرجاع اتصال MySQL أو None لو غير متاح."""
    if not mysql or not DB_USER or not DB_NAME:
        return None
    try:
        cn = mysql.connector.connect(
            host=DB_HOST, user=DB_USER, password=DB_PASS, database=DB_NAME, autocommit=True
        )
        return cn
    except Exception as e:
        print("MySQL connect error:", e)
        return None
# ====================================================

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/check", methods=["POST"])
def check():
    data = request.get_json(silent=True) or {}
    text = (data.get("text") or "").strip()
    if not text:
        return jsonify({"error": "empty", "message": "النص فارغ"}), 400

    results = []
    global_used = []  # لمنع تكرار نفس المدى عبر تداخل القطع

    def free_global(a, b):
        for g0, g1 in global_used:
            if not (b <= g0 or a >= g1):
                return False
        return True

    for base, chunk in chunk_text_with_overlap(text):
        matches = gemini_check_chunk(chunk)
        for m in matches:
            a = base + m["offset"]
            b = a + m["length"]
            # تجاهل أي مدى يخرج عن النص أو يتداخل مع مدى سبق اعتماده (بسبب الـ overlap)
            if 0 <= a < len(text) and b <= len(text) and free_global(a, b):
                global_used.append((a, b))
                results.append({
                    "offset": a,
                    "length": m["length"],
                    "message": m["message"],
                    "replacements": m.get("replacements", [])[:5],
                    "rule_id": m.get("rule_id", "GEMINI_25_FLASH_AR")
                })

    results.sort(key=lambda x: x["offset"])
    return jsonify({"matches": results})

# ========== NEW: مسار عدد المستخدمين ==========
@app.get("/user-count")
def user_count():
    """
    يرجّع عدد المستخدمين من جدول في MySQL بصيغة:
    {"count": <int>}
    لو تعذّر الاتصال أو الاستعلام → يرجّع 0 بهدوء.
    """
    cn = get_db()
    if not cn:
        return jsonify({"count": 0})
    try:
        cur = cn.cursor()
        tbl = DB_TABLE.replace("`", "")  # تبسيط
        cur.execute(f"SELECT COUNT(*) FROM `{tbl}`;")
        (count,) = cur.fetchone() or (0,)
        cur.close(); cn.close()
        return jsonify({"count": int(count or 0)})
    except Exception as e:
        print("user-count error:", e)
        try:
            cn.close()
        except Exception:
            pass
        return jsonify({"count": 0})
# =================================================
@app.post("/track")
def track():
    import uuid, mysql.connector, os
    uid = (request.json or {}).get("uid") or str(uuid.uuid4())
    cn = mysql.connector.connect(
        host=os.getenv("DB_HOST","127.0.0.1"),
        user=os.getenv("DB_USER","root"),
        password=os.getenv("DB_PASSWORD",""),
        database=os.getenv("DB_NAME","spelling_correcter"),
        autocommit=True
    )
    cur = cn.cursor()
    cur.execute(f"""
      INSERT INTO `{os.getenv('DB_TABLE','users')}` (uid, last_seen)
      VALUES (%s, NOW())
      ON DUPLICATE KEY UPDATE last_seen=VALUES(last_seen)
    """, (uid,))
    cur.close(); cn.close()
    return jsonify({"ok":True})

@app.get("/diag")
def diag():
    """
    تشخيص سريع: وجود المفتاح، اسم النموذج، وفحص قصير يرجّع matches.
    لا يطبع المفتاح نفسه.
    """
    info = {
        "env": {
            "GEMINI_API_KEY_present": bool(API_KEY),
            "MODEL_NAME": os.getenv("MODEL_NAME", ""),
        },
        "libs": {},
        "probe": {}
    }
    try:
        import google.generativeai as g
        info["libs"]["google-generativeai"] = getattr(g, "__version__", "unknown")
    except Exception as e:
        info["libs"]["google-generativeai"] = f"missing ({e})"
    try:
        from google import genai as g2
        info["libs"]["google-genai"] = getattr(g2, "__version__", "unknown")
    except Exception as e:
        info["libs"]["google-genai"] = f"missing ({e})"

    try:
        sample = "هاذا كتاب جميل"  # متعمد فيه خطأ "هاذا"
        raw = gen_gemini_response(sample)
        data = _extract_json(raw)
        ms = data.get("matches", [])
        info["probe"] = {"raw_len": len(raw or ""), "matches_len": len(ms)}
    except Exception as e:
        info["probe"] = {"error": str(e)}

    return jsonify(info)


if __name__ == "__main__":
    app.run(debug=True)
