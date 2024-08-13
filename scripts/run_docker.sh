#!/bin/bash
docker_image=tlbot/env:latest
# docker pull tlbot/env:latest
docker pull $docker_image

id=tlbot_latest

docker run \
    --name=$id\
    --rm \
    --interactive \
    --tty \
    --workdir $(pwd)\
    --hostname $(hostname) \
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
    --volume "$(pwd)/scripts/bashrc:$HOME/.bashrc:ro" \
    --volume "$HOME/.ccache:$HOME/.ccache:rw" \
    --tmpfs "$HOME:exec,rw,uid=$(id -u)" \
    --tmpfs "$HOME/.vscode-server:exec,rw,uid=$(id -u)" \
    --volume $(pwd):$(pwd) \
    --user $(id -u) \
    $docker_image
    bash --rcfile ~/.bashrc
