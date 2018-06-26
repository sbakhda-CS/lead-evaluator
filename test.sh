#!/usr/bin/env bash

docker run shiksb/mymodel:latest --train --context "$(cat test/train_input_message.json)"