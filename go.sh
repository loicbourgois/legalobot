#!/bin/sh
# $HOME/github.com/loicbourgois/legalobot/go.sh
path="legalobot"
name="legalobot"
path=$path name=$name docker-compose \
    --file $HOME/github.com/loicbourgois/$path/docker-compose.yml \
    up \
    --build --force-recreate \
    --exit-code-from $name \
    $name
