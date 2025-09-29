FROM python:3.11-slim

# إعدادات أساسية
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# نسخ المتطلبات وتثبيتها
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt gunicorn

# نسخ بقية الكود
COPY . .

# تشغيل Gunicorn على المنفذ 8080
CMD ["gunicorn", "-w", "2", "-k", "gthread", "-b", "0.0.0.0:8080", "app:app"]
