#!/bin/sh
set -ex
NAMESPACE=$MOUNT_NAMESPACE
VINEYARD_NAME=$VINEYARD_FULL_NAME
MARKER=$PRESTOP_MARKER
MOUNT_DIR="/runtime-mnt/vineyard/$NAMESPACE/$VINEYARD_NAME"
SOCKET_FILE="$MOUNT_DIR/vineyard-fuse/vineyard.sock"
FUSE_DIR="$MOUNT_DIR/vineyard-fuse"
RPC_CONFIG_FILE="$MOUNT_DIR/vineyard-fuse/rpc-conf/VINEYARD_RPC_ENDPOINT"
VINEYARD_YAML_FILE="$FUSE_DIR/vineyard.yaml"

# Write the IPCSocket and RPCEndpoints to the vineyard configurations YAML file
write_yaml_config() {
    echo "Vineyard:" > $VINEYARD_YAML_FILE
    echo "  IPCSocket: vineyard.sock" >> $VINEYARD_YAML_FILE
    echo "  RPCEndpoint: $1" >> $VINEYARD_YAML_FILE
}

mkdir -p $FUSE_DIR
while true; do
    # check if prestop marker exists, if so, skip mounting
    if [ -f $MARKER ]; then
        echo "PreStop hook is in progress, skip mounting."
        break
    fi
    # before mounting, store the rpc endpoint to a variable
    if [ -f $RPC_CONFIG_FILE ]; then
        VINEYARD_RPC_ENDPOINT=$(cat $RPC_CONFIG_FILE)
    else
        echo "rpc config file $RPC_CONFIG_FILE does not exist."
    fi

    if [ ! -S $SOCKET_FILE ]; then
        echo "Checking if vineyard-fuse is already a mount point..."
        if ! mountpoint -q $FUSE_DIR; then
            echo "mount vineyard socket..."
            mount --bind $MOUNT_DIR $FUSE_DIR
            echo "write vineyard ipc socket and rpc endpoint to vineyard configuration YAML..."
            write_yaml_config "$VINEYARD_RPC_ENDPOINT"
        else
            echo "$FUSE_DIR is already a mount point."
        fi
    else
        echo "$SOCKET_FILE exists."
    fi
    # wait for a minute so that the fuse mount point can be checked again
    sleep 60
done
