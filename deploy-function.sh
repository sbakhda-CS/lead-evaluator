#!/usr/bin/env bash
set -e

img="shiksb/lead-eval:$RANDOM"

docker build -t "$img" .
docker push "$img"

echo -e "\n\nRunning $img on docker...\n\n"
cat test/test_req.json | docker run --rm -p 9091:9091 -i $img ubuntu /bin/bash -c 'cat'

echo -e "\n\nRunning $img on cortex...\n\n"
cortex actions deploy --docker registry.cortex-develop.insights.ai:5000/$img le/lead_eval

echo -e "\n\nDeploying $img daemon on cortex...\n\n"
cortex actions deploy le/lead-inquire --actionType daemon --docker $img  --port "9091" --cmd '["--daemon"]'