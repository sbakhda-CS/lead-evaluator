#!/usr/bin/env bash

docker run shiksb/mymodel:latest --train --context "$(cat test/train_input_message.json)"
docker run shiksb/mymodel:latest --inquire --context "$(cat test/inquiry_input_message.json)"