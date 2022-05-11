#!/bin/bash
[ "$(git rev-parse HEAD)" = "$(git ls-remote origin | grep HEAD | awk '{print $1;}')" ]
