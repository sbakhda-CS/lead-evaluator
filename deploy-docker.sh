#!/usr/bin/env bash

IMAGE=shiksb/mymodel:latest
IMAGE_NAME=shiksb/mymodel
VERSION=(git describe --tags --dirty --always --long)

docker build -t ${IMAGE_NAME}:${VERSION} -f Dockerfile .
docker push ${IMAGE_NAME}:${VERSION}
docker tag ${IMAGE_NAME}:${VERSION} ${IMAGE_NAME}:latest
docker push ${IMAGE_NAME}:latest