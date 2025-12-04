#!/bin/bash

LOGFILE="/var/log/bot-autoupdate.log"

echo "===========================================" >> $LOGFILE
echo "[UPDATE] $(date '+%Y-%m-%d %H:%M:%S') Starting update..." >> $LOGFILE

# 1) Garantir que o diretório existe
if [ ! -d "/opt/bot" ]; then
    echo "[ERROR] /opt/bot não encontrado!" | tee -a $LOGFILE
    exit 1
fi

cd /opt/bot

# 2) Atualizar código
echo "[UPDATE] Running git pull..." >> $LOGFILE
git reset --hard >> $LOGFILE 2>&1
git pull >> $LOGFILE 2>&1

# 3) Garantir que o ambiente virtual existe
if [ ! -f "/opt/bot/venv/bin/pip" ]; then
    echo "[ERROR] Virtualenv não encontrado em /opt/bot/venv" | tee -a $LOGFILE
    exit 1
fi

# 4) Atualizar dependências
echo "[UPDATE] Installing Python dependencies..." >> $LOGFILE
/opt/bot/venv/bin/pip install -r requirements.txt >> $LOGFILE 2>&1

# 5) Reiniciar o serviço
echo "[UPDATE] Restarting service..." >> $LOGFILE
sudo systemctl restart binancebot.service

echo "[UPDATE] Bot updated and restarted successfully." | tee -a $LOGFILE
echo "===========================================" >> $LOGFILE
