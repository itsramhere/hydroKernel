#!/bin/bash
# Firecracker MicroVM Launch Script for Ephemeral Alert Dispatch Worker
# Phase 6: Hardware-Isolated Alert Formatting & Notification Fan-Out

set -e

SOCKET_PATH="/tmp/firecracker.socket"
KERNEL_PATH="./microvm/vmlinux"
ROOTFS_PATH="./microvm/rootfs.ext4"
TAP_DEV="tap0"
GUEST_IP="172.16.0.2"
HOST_IP="172.16.0.1"

echo "[Firecracker] Checking for /dev/kvm..."
if [ ! -e /dev/kvm ]; then
    echo "[Error] /dev/kvm is not accessible. Firecracker requires hardware virtualization." >&2
    exit 1
fi

# Clean up stale socket
rm -f "$SOCKET_PATH"

echo "[Firecracker] Starting firecracker daemon..."
firecracker --api-sock "$SOCKET_PATH" &
FC_PID=$!

# Wait for socket to be ready
while [ ! -S "$SOCKET_PATH" ]; do
    sleep 0.01
done

echo "[Firecracker] Configuring boot source..."
curl --unix-socket "$SOCKET_PATH" -i \
    -X PUT "http://localhost/boot-source" \
    -H "Accept: application/json" \
    -H "Content-Type: application/json" \
    -d "{
        \"kernel_image_path\": \"${KERNEL_PATH}\",
        \"boot_args\": \"console=ttyS0 reboot=k panic=1 pci=off ip=${GUEST_IP}::${HOST_IP}:255.255.255.0::eth0:off\"
    }"

echo "[Firecracker] Configuring root drive..."
curl --unix-socket "$SOCKET_PATH" -i \
    -X PUT "http://localhost/drives/rootfs" \
    -H "Accept: application/json" \
    -H "Content-Type: application/json" \
    -d "{
        \"drive_id\": \"rootfs\",
        \"path_on_host\": \"${ROOTFS_PATH}\",
        \"is_root_device\": true,
        \"is_read_only\": true
    }"

echo "[Firecracker] Configuring network interface (${TAP_DEV})..."
curl --unix-socket "$SOCKET_PATH" -i \
    -X PUT "http://localhost/network-interfaces/eth0" \
    -H "Accept: application/json" \
    -H "Content-Type: application/json" \
    -d "{
        \"iface_id\": \"eth0\",
        \"guest_mac\": \"AA:FC:00:00:00:01\",
        \"host_dev_name\": \"${TAP_DEV}\"
    }"

echo "[Firecracker] Issuing InstanceStart..."
curl --unix-socket "$SOCKET_PATH" -i \
    -X PUT "http://localhost/actions" \
    -H "Accept: application/json" \
    -H "Content-Type: application/json" \
    -d "{
        \"action_type\": \"InstanceStart\"
    }"

echo "[Firecracker] MicroVM booted successfully (PID: $FC_PID)."
