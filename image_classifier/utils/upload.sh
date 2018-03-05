#!/bin/bash

SERVER_HOST="chs-g1c-s-gpu-0.blupig.net"

cd "$(dirname "$0")"
ssh ${SERVER_HOST} 'rm -f ~/exp/*.py'
scp ./* ${SERVER_HOST}:~/exp/
