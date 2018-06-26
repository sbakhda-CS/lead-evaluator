#!/usr/bin/env bash
set -e
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

IMAGE=shiksb/mymodel:latest
IMAGE_NAME=shiksb/mymodel
VERSION=(git describe --tags --dirty --always --long)


cd ${SCRIPT_DIR}

# BUILD
echo "Building train function"
rm -f build/generic-train.zip
zip -j build/generic-train.zip train/*

# SAVE
echo "Creating job definition"
cortex jobs save -y training-job.yml

echo "Deploying function"
cortex actions deploy cortex/generic-train --kind nodejs:8 --code build/generic-train.zip

echo "Deploy daemon"
cortex actions deploy cortex/generic-inquire --actionType daemon --docker ${IMAGE} --port '9091' --cmd '["--daemon"]'


echo "Deploying Skill"
cortex skills save -y skill.yml

echo "Deploying Types"
cortex types save -y types.yml
cortex types save training-data-type.json


echo "Deploying Datasets"
cortex datasets save training-data-dataset.json

echo "Deploying training data"
cortex content upload mymodel/data.csv data.csv

echo "Deploying on Docker"
docker build -t ${IMAGE_NAME}:${VERSION} -f Dockerfile .
docker push ${IMAGE_NAME}:${VERSION}
docker tag ${IMAGE_NAME}:${VERSION} ${IMAGE_NAME}:latest
docker push ${IMAGE_NAME}:latest