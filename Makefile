TRAIN_INPUT_FILE := ./test/train_input_message.json
INQUIRY_INPUT_FILE := ./test/inquiry_input_message.json
TRAIN_INPUT := $(shell cat ${TRAIN_INPUT_FILE})
INQUIRY_INPUT := $(shell cat ${INQUIRY_INPUT_FILE})

    #CORTEX_URL = http://localhost:8000
CORTEX_URL = https://api.cortex-dev.insights.ai
    #INQUIRY_URL = https://realestate.cortex-stage.insights.ai
INQUIRY_URL = http://localhost:9091

VERSION := $(shell git describe --tags --dirty --always --long)
    #VERSION = cf3bfc9-vadan

IMAGE_NAME = shiksb/mymodel

bootstrap:
	./deploy-all.sh
	./test/python mk_test_input_messages.py

train:
	python model/main.py --train --context '${TRAIN_INPUT}' 

inquire_init:
	python model/main.py --inquire_init --context '${INQUIRY_INPUT}'

inquire:
	python model/main.py --inquire --context '${INQUIRY_INPUT}'

daemon:
	python model/main.py --daemon 

daemon.inquire: 
	curl -X POST ${INQUIRY_URL}/inquire --insecure -H "Content-Type: application/json" -d '${INQUIRY_INPUT}' 

get.serialized:
	curl -X GET ${CORTEX_URL}/v2/models/${MODEL_ID}/serialized -H "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJjb2duaXRpdmVzY2FsZS5jb20iLCJhdWQiOiJjb3J0ZXgiLCJzdWIiOiJuYm91bWF6YSIsInRlbmFudCI6ImNvZ3NjYWxlIiwiYWNsIjpbeyJuYW1lIjoiY29uc29sZSIsIm93bmVyIjoibmJvdW1hemEiLCJ0ZW5hbnQiOiJjb2dzY2FsZSIsImFjY2VzcyI6Im5vbmUiLCJvd25lclR5cGUiOiJ0ZW5hbnQifV0sImJlYXJlciI6InB1YmxpYyIsImtleSI6InRwTURmdTdjTTFjMmtNVnhDQTI2b1czUzc2UmN5eU1zIiwiZXhwIjoxNTIwMDAxNzc5LCJpYXQiOjE1MTg3OTIxNzl9.p4lrVwLjAXevkMyGU7PP2sdWnoH_mLn7X_ewjpUmkOs" | jq .

# Docker stuff

docker.build:
	docker build -t ${IMAGE_NAME}:${VERSION} -f Dockerfile .

docker.train:
	docker run --rm ${IMAGE_NAME}:${VERSION} --train --context '${TRAIN_INPUT}'

docker.inquire_init:
	docker run --rm ${IMAGE_NAME}:${VERSION} --inquire_init --context '${INQUIRY_INPUT}'

docker.inquire:
	docker run --rm ${IMAGE_NAME}:${VERSION} --inquire --context '${INQUIRY_INPUT}'

docker.daemon:
	docker run --rm -p 9091:9091 ${IMAGE_NAME}:${VERSION} --daemon 

docker.push:
	docker push ${IMAGE_NAME}:${VERSION} 
	docker tag ${IMAGE_NAME}:${VERSION} ${IMAGE_NAME}:latest
	docker push ${IMAGE_NAME}:latest


docker.deploy: docker.build docker.push 
