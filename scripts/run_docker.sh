#!/bin/bash


docker_image=tlbot/env:latest
$(docker pull $docker_image)

id=tlbot_latest
# Attaches to `id` container if it already started.
if [ ! -z "$(docker ps --filter name=$id | grep $id)" ]; then
    echo "Attach to existing Docker container [$id]"
    docker exec --interactive --tty $id /bin/bash
    exit 0  
fi

docker run \
    --name=$id\
    --rm \
    --interactive \
    --tty \
    --gpus all\
    --workdir $(pwd)\
    --volume "/run/user:/run/user" \
    --volume "/tmp:/tmp" \
    --volume "/dev:/dev" \
    --volume "$HOME/.ssh:$HOME/.ssh" \
    --volume "/etc/localtime:/etc/localtime:ro" \
    --volume "/etc/passwd:/etc/passwd:ro" \
    --volume "/etc/shadow:/etc/shadow:ro" \
    --volume "/etc/group:/etc/group:ro" \
    --volume "/etc/gshadow:/etc/gshadow:ro" \
    --volume "/etc/apt/apt.conf:/etc/apt/apt.conf:ro" \
    --volume "$HOME/.cache:$HOME/.cache:rw" \
    --volume "$pwd/scripts/bashrc:$HOME/.bashrc:ro" \
    --volume "$HOME/.ccache:$HOME/.ccache:rw" \
    --tmpfs "$HOME:exec,rw,uid=$(id -u)" \
    --tmpfs "$HOME/.vscode-server:exec,rw,uid=$(id -u)" \
    --volume $pwd:$pwd \
    --user $(id -u) \
    $docker_image
    bash --rcfile ~/.bashrc
