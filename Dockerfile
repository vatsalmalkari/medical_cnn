# Build stage
FROM python:3.10-slim-buster as builder

WORKDIR /app
COPY requirements.txt .
RUN pip install --user -r requirements.txt

# Runtime stage
FROM python:3.10-slim-buster

WORKDIR /app
COPY --from=builder /root/.local /root/.local
COPY . .

ENV PATH=/root/.local/bin:$PATH
ENV PYTHONPATH=/app
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]