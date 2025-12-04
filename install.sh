#!/bin/bash

set -e
echo "[BOT INSTALL] Starting installation..."

sudo dnf install -y git python3 python3-pip

# Atualizar diretório
sudo mkdir -p /opt/bot
sudo chmod -R 777 /opt/bot

# Se /opt/bot estiver vazio clonamos, senão só damos git pull
if [ ! -d "/opt/bot/.git" ]; then
    git clone https://github.com/Xuabo/Bot--Binance.git /opt/bot
else
    cd /opt/bot
    git pull
fi

# Criar ambiente virtual
python3 -m pip install --upgrade pip
python3 -m pip install virtualenv
python3 -m virtualenv /opt/bot/venv
/opt/bot/venv/bin/pip install -r /opt/bot/requirements.txt

echo "[BOT INSTALL] Installation complete."
