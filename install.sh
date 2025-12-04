#!/bin/bash

set -e
echo "[BOT INSTALL] Starting installation..."

##############################################
# 1) Instalar dependências do sistema
##############################################
sudo dnf install -y git wget gcc make openssl-devel bzip2-devel libffi-devel zlib-devel sqlite-devel python3 python3-pip


##############################################
# 2) Instalar pyenv + Python 3.8.18
##############################################
if [ ! -d "$HOME/.pyenv" ]; then
    git clone https://github.com/pyenv/pyenv.git ~/.pyenv
    echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
    echo 'export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
    echo 'eval "$(pyenv init -)"' >> ~/.bashrc
    export PYENV_ROOT="$HOME/.pyenv"
    export PATH="$PYENV_ROOT/bin:$PATH"
    eval "$(pyenv init -)"
fi

if ! pyenv versions | grep -q "3.8.18"; then
    echo "[BOT INSTALL] Installing Python 3.8.18..."
    pyenv install 3.8.18
fi

pyenv global 3.8.18


##############################################
# 3) Preparar diretório do bot
##############################################
sudo mkdir -p /opt/bot
sudo chmod -R 777 /opt/bot


##############################################
# 4) Clonar ou atualizar repositório
##############################################
if [ ! -d "/opt/bot/.git" ]; then
    git clone https://github.com/Xuabo/Bot--Binance.git /opt/bot
else
    cd /opt/bot
    git pull
fi


##############################################
# 5) Criar ambiente virtual com Python 3.8.18
##############################################
python3 -m pip install --upgrade pip
python3 -m pip install virtualenv

python3 -m virtualenv /opt/bot/venv --python=$HOME/.pyenv/versions/3.8.18/bin/python3


##############################################
# 6) Instalar dependências do requirements.txt
##############################################
/opt/bot/venv/bin/pip install -r /opt/bot/requirements.txt
/opt/bot/venv/bin/pip install python-dotenv


##############################################
# 7) Criar serviço systemd do bot
##############################################
sudo bash -c 'cat <<EOF >/etc/systemd/system/binancebot.service
[Unit]
Description=Binance Trading Bot
After=network.target

[Service]
Type=simple
User=opc
WorkingDirectory=/opt/bot
EnvironmentFile=/opt/bot/.env
ExecStart=/opt/bot/venv/bin/python /opt/bot/ultra_precision_realtime_bot_binance.py
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF'


sudo systemctl daemon-reload
sudo systemctl enable binancebot.service

echo ""
echo "[BOT INSTALL] Installation complete."
echo "Agora crie o ficheiro /opt/bot/.env com as chaves da Binance."
echo ""

