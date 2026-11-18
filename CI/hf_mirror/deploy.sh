#!/bin/bash
ECR_REPO=369469875935.dkr.ecr.us-east-1.amazonaws.com
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin $ECR_REPO
docker build -t autoapex-hf-mirror .
docker tag autoapex-hf-mirror:latest $ECR_REPO/autoapex-hf-mirror:latest
docker push $ECR_REPO/autoapex-hf-mirror:latest