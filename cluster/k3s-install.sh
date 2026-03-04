#!/bin/bash
# Bootstrap single-node k3s cluster bound to localhost

set -e

echo "Installing k3s..."
curl -sfL https://get.k3s.io | INSTALL_K3S_EXEC="--bind-address=127.0.0.1" sh -

echo "Waiting for node to be ready..."
kubectl wait --for=condition=Ready nodes --all --timeout=60s

echo "Applying cluster manifests..."
kubectl apply -f "$(dirname "$0")/manifests/"