#!/bin/bash

CUR_DIR=$(dirname "$0")

# Stop programs
echo "Stopping programs"

# Kill updater
if pgrep -f "$CUR_DIR/updater.py" > /dev/null; then
    echo "Updater is running, restarting"
    pkill -f "$CUR_DIR/updater.py"
fi

# Kill Watchtower
if docker ps -q -f name=watchtower; then
    echo "Watchtower is running, restarting"
    docker stop watchtower
    docker rm watchtower
fi

# Kill tastyfreshpos
if docker ps -q -f name=tastyfreshpos; then
    echo "tastyfreshpos is running, restarting"
    docker stop tastyfreshpos
    docker rm tastyfreshpos
fi

# Start programs
echo "starting tastyfreshpos"
docker run \
    --runtime nvidia \
    --network host \
    --device /dev/video0:/dev/video0 \
    --restart always \
    -d \
    -e DISPLAY:$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v /tmp/argus_socket:/tmp/argus_socket \
    --name tastyfreshpos \
    tastyfreshpos/tastyfreshpos:latest

# Start Watchtower
echo "Starting Watchtower"
docker run \
    --name watchtower \
    --network host \
    --restart always \
    -d \
    -v /var/run/docker.sock:/var/run/docker.sock \
    -e REPO_USER=tastyfreshpos \
    -e REPO_PASS=TastyFreshIIP2024 \
    -e WATCHTOWER_NO_STARTUP_MESSAGE=true \
    -e WATCHTOWER_CLEANUP=false \
    containrrr/watchtower \
    --interval 10 \
    tastyfreshpos

# Start updater Python script
echo "Starting Updater Script"
python3 "$CUR_DIR/updater.py"