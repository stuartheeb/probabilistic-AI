docker build --tag task2 C:\Users\nicos\eth-probabilistic-AI\projects\task2
docker run --rm --gpus all -u $(id -u):$(id -g) -v "$(pwd):/results" task2