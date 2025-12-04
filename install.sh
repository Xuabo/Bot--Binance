#!/bin/bash
set -e

echo "========================"
echo " Binance Bot Installer"
echo "========================"

LOG="/var/log/binancebot-install.log"
echo "[START] $(date)" > $LOG

########################################
# 1) CREATE SWAP (AVOID OOM KILLED)
########################################
echo "[1/10] Creating swap..." | tee -a $LOG
if [ ! -f /swapfile ]; then
    sudo fallocate -l 2G /swapfile
    sudo chmod 600 /swapfile
    sudo mkswap /swapfile
    sudo swapon /swapfile
    echo "/swapfile swap swap defaults 0 0" | sudo tee -a /etc/fstab
else
    echo "Swap already exists, skipping." | tee -a $LOG
fi

########################################
# 2) UPDATE SYSTEM
########################################
echo "[2/10] Updating system..." | tee -a $LOG
sudo apt update -y >> $LOG 2>&1
sudo apt install -y git wget build-essential libssl-dev libbz2-dev libffi-dev libsqlite3-dev zlib1g-dev python3 python3-pip curl >> $LOG 2>&1

########################################
# 3) INSTALL PYENV
########################################
echo "[3/10] Installing pyenv..." | tee -a $LOG
if [ ! -d "$HOME/.pyenv" ]; then
    git clone https://github.com/pyenv/pyenv.git ~/.pyenv >> $LOG 2>&1
    echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
    echo 'export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
    echo 'eval "$(pyenv init -)"' >> ~/.bashrc
fi

export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"

########################################
# 4) INSTALL PYTHON 3.8.18
########################################
echo "[4/10] Installing Python 3.8.18..." | tee -a $LOG
pyenv install -s 3.8.18 >> $LOG 2>&1
pyenv global 3.8.18

########################################
# 5) PREPARE BOT FOLDER
########################################
echo "[5/10] Preparing /opt/bot..." | tee -a $LOG
sudo mkdir -p /opt/bot
sudo chmod -R 777 /opt/bot

########################################
# 6) CLONE OR UPDATE BOT
########################################
echo "[6/10] Fetching bot repository..." | tee -a $LOG
if [ ! -d "/opt/bot/.git" ]; then
    git clone https://github.com/Xuabo/Bot--Binance.git /opt/bot >> $LOG 2>&1
else
    cd /opt/bot
    git pull >> $LOG 2>&1
fi

########################################
# 7) CREATE VENV
########################################
echo "[7/10] Creating virtualenv..." | tee -a $LOG
pip install virtualenv >> $LOG 2>&1
python3 -m virtualenv /opt/bot/venv --python="$HOME/.pyenv/versions/3.8.18/bin/python3" >> $LOG 2>&1

########################################
# 8) INSTALL REQUIREMENTS
########################################
echo "[8/10] Installing requirements..." | tee -a $LOG
/opt/bot/venv/bin/pip install --no-cache-dir -r /opt/bot/requirements.txt >> $LOG 2>&1

########################################
# 9) CREATE UPDATE.SH
########################################
echo "[9/10] Creating update script..." | tee -a $LOG

cat <<'EOF' >/opt/bot/update.sh
#!/bin/bash
LOGFILE="/var/log/bot-autoupdate.log"
echo "[UPDATE] $(date)" >> $LOGFILE

cd /opt/bot
git reset --hard >> $LOGFILE 2>&1
git pull >> $LOGFILE 2>&1
/opt/bot/venv/bin/pip install -r requirements.txt >> $LOGFILE 2>&1
sudo systemctl restart binancebot.service

echo "[DONE] $(date)" >> $LOGFILE
EOF

sudo chmod +x /opt/bot/update.sh

########################################
# 10) CREATE SYSTEMD SERVICE
########################################
echo "[10/10] Creating systemd service..." | tee -a $LOG

sudo bash -c 'cat <<EOF >/etc/systemd/system/binancebot.service
[Unit]
Description=Binance Trading Bot
After=network.target

[Service]
User=ubuntu
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

echo "=============================================="
echo " INSTALLATION COMPLETE!"
echo ""
echo " → Create /opt/bot/.env with:"
echo "     BINANCE_API_KEY=xxxx"
echo "     BINANCE_SECRET_KEY=xxxx"
echo ""
echo " → Start bot:"
echo "     sudo systemctl start binancebot"
echo ""
echo " → Monitor logs:"
echo "     journalctl -u binancebot -f"
echo "=============================================="

echo "[DONE] $(date)" >> $LOG


