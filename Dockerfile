FROM python:3.13-slim

WORKDIR /app
COPY pyproject.toml README.md /app/
COPY vllm_sleeper_proxy /app/vllm_sleeper_proxy
RUN pip install --no-cache-dir .

ENV SLEEPER_PROXY_HOST=0.0.0.0 \
    SLEEPER_PROXY_PORT=8889 \
    PYTHONUNBUFFERED=1

EXPOSE 8889
CMD ["vllm-sleeper-proxy"]
