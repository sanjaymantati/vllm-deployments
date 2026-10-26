ARG VLLM_VERSION=latest
FROM vllm/vllm-openai:${VLLM_VERSION}

# Image already sets the OpenAI server entrypoint; args come from compose `command`.
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=5s --start-period=180s --retries=40 \
  CMD python3 -c "import urllib.request as u; u.urlopen('http://localhost:8000/health', timeout=3)" || exit 1
