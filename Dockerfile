FROM python:3.9-slim

WORKDIR /app

RUN apt-get update && apt-get install -y gcc libpq-dev

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8020

CMD ["sh", "-c", "alembic stamp head && alembic revision --autogenerate -m 'initial migration' && alembic upgrade head && uvicorn main:app --host 0.0.0.0 --port 8020 --reload"]
