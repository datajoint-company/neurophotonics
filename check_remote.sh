#!/bin/bash
 
str1=$(git remote show origin | grep pushes\ to\ master)
str2="    master    pushes to master    (up to date)"
 
if [ "$str1" != "$str2" ]; then
    echo "Both Strings are not Equal."
else
    echo "Both Strings are Equal."
fi
