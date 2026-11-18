FROM python:3.13-slim

WORKDIR /app

COPY backend/requirements.txt ./requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

COPY backend/main.py backend/init_db.py ./backend/

COPY ml/main.py ./ml/

COPY data/raw/ ./data/raw/

COPY data/processed/courses.csv ./data/processed/courses.csv

RUN python backend/init_db.py

EXPOSE 8000

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
