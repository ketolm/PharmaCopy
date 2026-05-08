FROM python:3.14-slim

WORKDIR /app/app/src

# Copy project files into the container
COPY pyproject.toml /app/
COPY app /app/app
COPY README.md /app/README.md

RUN mkdir -p /app/app/src/data /app/hf_cache
RUN python -m pip install --no-cache-dir pip setuptools wheel uv
RUN uv pip install --system /app

ENV PYTHONPATH=/app/app/src
ENV HF_HOME=/app/hf_cache

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
