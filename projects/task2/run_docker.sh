# try to enable gpu support
docker build --tag task2 .
#docker run --rm -u $(id -u):$(id -g) -v "$(pwd):/results" task2
docker run --rm --gpus all -u $(id -u):$(id -g) -v "$(pwd):/results" task2