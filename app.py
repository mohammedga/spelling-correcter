import os, re, json
from flask import Flask, render_template, request, jsonify

# حمّل متغيّرات البيئة من .env محليًا
from dotenv import load_dotenv
load_dotenv()

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
PROMPT_TMPL = """أنت مدقق إملائي/نحوي عربي.
أعد JSON فقط بهذه البنية الصارمة:
{{
  "matches": [
    {{
      "offset": number,              // موضع بداية الخطأ داخل النص المُعطى بين <<< >>>
      "length": number,              // طول المقطع الخاطئ
      "surface": string,             // المقطع الخاطئ كما هو حرفيًا
      "message": string,
      "replacements": [string, ...], // حتى 5 اقتراحات
      "rule_id": "GEMINI_25_FLASH_AR"
    }}
  ]
}}
- إن لم توجد أخطاء: {{"matches": []}}
- لا تعد نصًا مصححًا كاملًا.

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
    """محاولة مرنة لاستخراج JSON حتى لو النموذج أضاف نصًا زائدًا."""
    try:
        return json.loads(raw)
    except Exception:
        m = re.search(r"\{[\s\S]*\}", raw)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                return {}
        return {}

def gen_gemini_response(snippet: str) -> str:
    """ينادي Gemini 2.5 Flash ويُعيد نص JSON خام."""
    # المحاولة 1: SDK الجديد google-genai
    try:
        from google import genai
        from google.genai import types
        client = genai.Client(api_key=API_KEY)
        cfg = types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0,
            # thinking_config=types.ThinkingConfig(thinking_budget=0),  # اختياري
        )
        resp = client.models.generate_content(
            model=MODEL_NAME,
            contents=build_prompt(snippet),
            config=cfg,
        )
        return resp.text or "{}"
    except ModuleNotFoundError:
        pass
    except Exception as e:
        print("genai(new) error:", e)

    # المحاولة 2: SDK القديم google-generativeai
    try:
        import google.generativeai as genai_old
        genai_old.configure(api_key=API_KEY)
        model = genai_old.GenerativeModel(MODEL_NAME)
        resp = model.generate_content(
            build_prompt(snippet),
            generation_config={"response_mime_type": "application/json", "temperature": 0}
        )
        return getattr(resp, "text", None) or "{}"
    except Exception as e:
        print("genai(old) error:", e)
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

if __name__ == "__main__":
    app.run(debug=True)


