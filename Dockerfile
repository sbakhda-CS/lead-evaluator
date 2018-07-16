FROM c12e/cortex-python-lib:latest-master

MAINTAINER CognitiveScale.com

RUN pip install xgboost matplotlib lime

# Set up the program in the image
COPY model /opt/program