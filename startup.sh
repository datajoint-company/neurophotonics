#!/bin/sh

echo "Setting up..."
git checkout master
docker-compose -f ./docker/docker-compose-dev.yaml down
git pull origin master

echo "Starting run..."
start_time=$(date +"%Y-%m-%d_%H:%M:%S")
WORKER_COUNT=1 HOST_UID=$(id -u) docker-compose -f ./docker/docker-compose-dev.yaml up --build

echo "Saving logs..."
echo "$(podman-compose -f ./docker/docker-compose-dev.yaml logs)" > log_${start_time}.txt
git checkout logs
git add log_${start_time}.txt
git commit -m "Add log for run on ${start_time}"
git push origin logs

echo "Ending run..."
git checkout master
docker-compose -f ./docker/docker-compose-dev.yaml down

echo "Shutting down..."
sudo poweroff
