# try to enable gpu support
if [ -f ./results_check.byte ]; then
  mv results_check.byte old_checkfiles/$(date +%m-%d_%H-%M -r results_check.byte)_results_check.byte
fi

docker build --tag task2 .
#docker run --rm -u $(id -u):$(id -g) -v "$(pwd):/results" task2
docker run --rm --gpus all -u $(id -u):$(id -g) -v "$(pwd):/results" task2
