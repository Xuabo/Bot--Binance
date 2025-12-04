#!/bin/bash

cd /opt/bot
git pull
/opt/bot/venv/bin/pip install -r requirements.txt
sudo systemctl restart binancebot.service
echo "[UPDATE] Bot updated and restarted."
