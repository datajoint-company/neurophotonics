#!/bin/sh

echo "Setting up..."
git checkout master
podman-compose -f ./docker/docker-compose-dev.yaml down
start_time=$(date +"%Y-%m-%d_%H:%M:%S")

echo "Starting run..."
until [ "$(git rev-parse HEAD)" = "$(git ls-remote origin | grep HEAD | awk '{print $1;}')" \
       && $(bash check_db.sh) = '0']
do
  podman-compose -f ./docker/docker-compose-dev.yaml down
  git pull origin master
  WORKER_COUNT=1 HOST_UID=$(id -u) podman-compose -f ./docker/docker-compose-dev.yaml up --build \
    >> log_${start_time}.txt 2>&1
done

echo "Saving logs..."
git checkout logs
git add log_${start_time}.txt
git commit -m "Add log for run on ${start_time}"
git push origin logs

echo "Ending run..."
git checkout master
podman-compose -f ./docker/docker-compose-dev.yaml down

echo "Shutting down..."
sudo poweroff
