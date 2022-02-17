#!/usr/bin bash
set -e

chmod -R a+rw /app/models
chmod -R a+rw /app/logs

export PYTHONPATH="${PYTHONPATH}:/app/" 

exec "$@"