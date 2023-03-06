FROM python:3.10
RUN apt-get update && apt-get install beep
COPY  requirements.txt requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip python -m pip install -r requirements.txt
