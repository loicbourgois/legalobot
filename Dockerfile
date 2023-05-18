FROM python:3.10
RUN apt-get update && apt-get install beep
COPY  requirements.txt requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip python -m pip install -r requirements.txt
RUN python3 -c "from sentence_transformers import SentenceTransformer ; model = SentenceTransformer('multi-qa-mpnet-base-cos-v1')" 
RUN python3 -c "from sentence_transformers import SentenceTransformer ; model = SentenceTransformer('distiluse-base-multilingual-cased-v2')" 
RUN python3 -c "from sentence_transformers import SentenceTransformer ; model = SentenceTransformer('sentence-transformers/gtr-t5-xl')" 
