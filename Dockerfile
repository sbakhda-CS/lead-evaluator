FROM frolvlad/alpine-python3:latest

MAINTAINER CognitiveScale.com

WORKDIR /opt/program
COPY model /opt/program

EXPOSE 80 9091 8888 5000

RUN pip install cortex-client

ENTRYPOINT ["python", "func.py"]