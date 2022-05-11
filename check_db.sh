keys=0
podman-compose -f ./docker/docker-compose-dev.yaml logs 2> /dev/null |\
  grep -E '^keys\s+processed:\s*[0-9]+$' |\
  while read line ; 
    do keys=$(( $keys + ${line:16} )) ; 
  done
echo $keys
