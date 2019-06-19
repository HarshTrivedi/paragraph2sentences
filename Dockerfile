FROM python:3.7.2
WORKDIR /run
COPY fftqdm.py .
COPY hotpotqa_para2sentences.py .
COPY hotpotqa_para2sentences_dirk.py .
COPY coref_replace_dirk.py .
COPY requirements.txt .
RUN pip install -r requirements.txt
RUN python -m spacy download en_core_web_md

ENTRYPOINT []
CMD ["/bin/bash"]
