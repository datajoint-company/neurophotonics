#!/bin/sh

echo "Setting up..."
git checkout master
podman system reset -f
start_time=$(date +"%Y-%m-%d_%H:%M:%S")

echo "Starting run..."
WORKER_COUNT=1 HOST_UID=$(id -u) podman-compose -f ./docker/docker-compose_standard_worker.yaml -p neurophotonics_standard up --build >> log_${start_time}.txt 2>&1
until [ "$(git rev-parse HEAD)" = "$(git ls-remote origin | grep HEAD | awk '{print $1;}')" ] && [ $(bash check_db.sh) = '0' ]
do
	echo "Additional work detected"
	echo
	echo "Restarting workflow..."
	podman system reset -f
	git pull origin master
	WORKER_COUNT=1 HOST_UID=$(id -u) podman-compose -f ./docker/docker-compose_standard_worker.yaml -p neurophotonics_standard up --build >> log_${start_time}.txt 2>&1
	done
	echo "Workflow complete"

	echo
	echo "Saving logs..."
	git checkout logs
	git add log_${start_time}.txt
	git commit -m "Add log for run on ${start_time}"
	git push origin logs
	echo "Logs saved"

	echo
	echo "Ending run..."
	git checkout master
	podman system reset -f

	echo
	echo "Shutting down..."
	sudo poweroff
