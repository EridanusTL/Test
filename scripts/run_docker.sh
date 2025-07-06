#!/bin/bash

docker_image=tlbot/ubuntu22-robotics:latest
if branch_name=$(git rev-parse --abbrev-ref HEAD 2>/dev/null); then
    id="$branch_name"
else
    echo "Warning: Not in git , use default ID 'ros-humble'"
    id=ros-humble 
fi

docker pull $docker_image

for gid in $(id -G); do
  group_add_opts="$group_add_opts --group-add $gid"
done

if [ "$(docker ps -q --filter "name=^$id$")" ]; then
    echo "Container $id is already running."    
    echo "Attach on Container $id."
    docker exec -it $id bash --rcfile ~/.bashrc

else
    if [ "$(docker ps -q --filter "name=^$id$")" ]; then
        echo "Starting existing container $id."
        docker start $id
        docker exec -it $id bash --rcfile ~/.bashrc
    else
        echo "Creating and starting new container $id."
        docker run \
            --network host \
            --privileged \
            --name=$id \
            --rm \
            --interactive \
            --tty \
            --workdir $(pwd) \
            --hostname $(hostname) \
            --gpus all \
            --env "DISPLAY=$DISPLAY" \
            --env "QT_X11_NO_MITSHM=1" \
            -e NVIDIA_DRIVER_CAPABILITIES=all \
            --env="WORKSPACE_PATH=$(pwd)" \
            --volume "/tmp/.X11-unix:/tmp/.X11-unix:rw" \
            --volume "/run/user:/run/user" \
            --volume "/tmp:/tmp" \
            --volume "/dev:/dev" \
            --volume "/etc/localtime:/etc/localtime:ro" \
            --volume "/etc/passwd:/etc/passwd:ro" \
            --volume "/etc/shadow:/etc/shadow:ro" \
            --volume "/etc/group:/etc/group:ro" \
            --volume "/etc/gshadow:/etc/gshadow:ro" \
            --volume "/etc/apt/apt.conf:/etc/apt/apt.conf:ro" \
            --volume "$(pwd):$(pwd)" \
            --volume "$(pwd)/scripts/bashrc:$HOME/.bashrc" \
            --volume "$(pwd)/data:$(pwd)/data:rw" \
            --volume "$HOME/.ssh:$HOME/.ssh" \
            --volume "$HOME/.cache:$HOME/.cache:rw" \
            --volume "$HOME/.ccache:$HOME/.ccache:rw" \
            --volume "$HOME/.vscode-server/extensions:$HOME/.vscode-server/extensions:rw" \
            --tmpfs "$HOME/.vscode-server:exec,rw,uid=$(id -u)" \
            --tmpfs "$HOME:exec,rw,uid=$(id -u)" \
            --user  $(id -u)\
            $group_add_opts \
            $docker_image \
            bash --rcfile ~/.bashrc
    fi
fi


