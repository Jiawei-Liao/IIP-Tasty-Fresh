# Usage
1. Open terminal 1
2. Run `cd ./detect/POS/backend; uvicorn main:app`
3. Open terminal 2
4. Run `cd ./detect/POS/frontend; npm run live`
For NVIDIA Jetson, run:
docker build -t pos:latest .
docker run --runtime nvidia --network host --device /dev/video0:/dev/video0 -e DISPLAY:$DISPLAY -v /tmp/.X11-unix:/tmp.X11-unix -v /tmp/argus_socket:/tmp/argus_socket pos:latest

# Watchtower
1. Run `pip install -r watchtower/requirements.txt`
1. Run `docker login` and input credentials
2. Run `chmod +x start.sh`
3. Run `./start.sh`
