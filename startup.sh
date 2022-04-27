#! /bin/bash

echo "Setting up..."
podman-compose -f ./docker/docker-compose-dev.yaml down
git checkout master 
git pull origin master
date=$( date '+%Y-%m-%d_%H-%M-%S' )

echo "Starting workflow..."
WORKER_COUNT=1 HOST_UID=$(id -u) podman-compose -f ./docker/docker-compose-dev.yaml up --build --exit-code-from standard_worker

echo "Saving logs..."
podman-compose -f ./docker/docker-compose-dev.yaml logs > log_$date.txt 2>&1
git checkout logs
git add log_$date.txt
git commit -m "log file for run at $date"
git push origin logs

echo "Shutting down..."
git checkout master
podman-compose -f ./docker/docker-compose-dev.yaml down
sudo shutdown -h now
