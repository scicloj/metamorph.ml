#!/bin/sh
poetry lock && poetry install --sync --no-root && poetry lock && poetry run clj -M:nrepl
