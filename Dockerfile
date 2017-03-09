FROM tensorflow/tensorflow:0.12.1

RUN pip install -U pip && \
    pip install 'keras==1.2.1' && \
    pip install 'h5py==2.6.0' && \
    pip install 'spacy==1.6.0'

RUN python -m spacy.en.download all && \
    python -m spacy.de.download all

RUN pip install 'Flask==0.10.1'

COPY intent/ /intent/

WORKDIR /
