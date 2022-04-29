#!/bin/bash
 
str1=$(git ls-remote origin | grep HEAD | awk '{print $1;}')
str2=$(git rev-parse HEAD)
 
if [ "$str1" != "$str2" ]; then
    echo "False"
else
    echo "True"
fi
