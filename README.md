To run the cluster build script also argocd 

First the cluster after the argo cd 

chmod +x bootstrap.sh 

./bootstrap.sh 

And after running both create the root app 

kubectl apply -f argocd/root-app.yaml 
