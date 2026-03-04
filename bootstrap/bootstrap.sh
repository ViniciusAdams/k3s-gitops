#!/bin/bash
set -e

echo "🔹 Creating Argo CD namespace..."
kubectl apply -f argo-cd/namespace.yaml

echo "🔹 Deploying Argo CD..."
kubectl apply -n argocd -f https://raw.githubusercontent.com/argoproj/argo-cd/v2.9.8/manifests/install.yaml

echo "🔹 Waiting for Argo CD pods to be ready..."
kubectl wait --for=condition=Ready pods -n argocd --all --timeout=180s

echo "Argo CD should be up and running!"
echo "Access Argo CD server (port-forward):"
echo "kubectl port-forward svc/argocd-server -n argocd 8080:443"