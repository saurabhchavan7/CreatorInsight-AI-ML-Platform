#!/bin/bash

# Log everything for debugging
exec > /home/ubuntu/start_docker.log 2>&1

echo "Starting deployment at $(date)"

# -----------------------------
# Variables (easy to change)
# -----------------------------
AWS_REGION="us-east-1"
ECR_REGISTRY="992382586780.dkr.ecr.us-east-1.amazonaws.com"
IMAGE_NAME="creatorinsight-ml-api"
IMAGE_TAG="latest"
CONTAINER_NAME="creatorinsight-api"

# -----------------------------
# Login to ECR
# -----------------------------
echo "Logging in to ECR..."
aws ecr get-login-password --region $AWS_REGION \
| docker login --username AWS --password-stdin $ECR_REGISTRY

# -----------------------------
# Pull latest image
# -----------------------------
echo "Pulling Docker image..."
docker pull $ECR_REGISTRY/$IMAGE_NAME:$IMAGE_TAG

# -----------------------------
# Stop & remove old container
# -----------------------------
if [ "$(docker ps -q -f name=$CONTAINER_NAME)" ]; then
    echo "Stopping existing container..."
    docker stop $CONTAINER_NAME
fi

if [ "$(docker ps -aq -f name=$CONTAINER_NAME)" ]; then
    echo "Removing existing container..."
    docker rm $CONTAINER_NAME
fi

# -----------------------------
# Run new container
# -----------------------------
echo "Starting new container..."
docker run -d \
  --name $CONTAINER_NAME \
  -p 80:5000 \
  -e MLFLOW_TRACKING_URI=http://ec2-3-84-182-30.compute-1.amazonaws.com:5000/ \
  $ECR_REGISTRY/$IMAGE_NAME:$IMAGE_TAG

echo "Deployment completed successfully at $(date)"
