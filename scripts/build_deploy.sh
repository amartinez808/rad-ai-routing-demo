#!/usr/bin/env bash
set -e
docker compose build
docker compose up -d
docker compose ps
echo "App should be reachable via the nginx service on port 80 of the host."
