#!/bin/bash
for i in {1..2500}; do
    python run_ppo_drone.py
    echo "'train_agent.py' crashed with exit code $?. Restarting..." >&2
    sleep 30
done
