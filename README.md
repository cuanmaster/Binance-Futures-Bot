# Binance Futures Trading Bot 🤖

Bot trading otomatis untuk Binance Futures dengan integrasi AI Gemini dan notifikasi Telegram. Bot ini melakukan analisis teknikal multi-pair dan menggunakan AI untuk konfirmasi sinyal trading.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Version](https://img.shields.io/badge/Version-1.0.0-orange.svg)

## ✨ Fitur Utama

- 🔄 **Multi-Pair Trading**: Support multiple cryptocurrency pairs secara simultan
- 🤖 **AI-Powered Analysis**: Integrasi dengan Gemini AI untuk analisis market
- 📊 **Technical Analysis**: Indikator teknikal lengkap (RSI, MACD, Bollinger Bands, EMA, dll)
- 📱 **Telegram Notifications**: Notifikasi real-time via Telegram
- ⚡ **Risk Management**: Management risiko dengan stop loss dan take profit otomatis
- 📈 **Performance Tracking**: Statistik trading dan laporan performa
- 🔒 **Environment Security**: Konfigurasi aman menggunakan environment variables

## 🛠️ Teknologi

- **Python 3.8+**
- **CCXT** - Crypto Exchange Trading Library
- **TA-Lib** - Technical Analysis Library
- **Google Gemini AI** - Artificial Intelligence
- **python-telegram-bot** - Telegram Bot Integration
- **pandas & numpy** - Data Analysis

## 📋 Prerequisites

- Akun Binance dengan Futures trading enabled
- API Keys dari Binance
- API Key dari Google AI Studio (Gemini)
- Bot Telegram (dari BotFather)

## 🚀 Instalasi Cepat

### 1. Clone & Setup

```bash
# Clone repository
git clone https://github.com/yourusername/binance-futures-bot.git
cd binance-futures-bot

# Install dependencies
pip install -r requirements.txt

# Install TA-Lib (Windows)
pip install talib-binance

# Atau untuk Linux/Mac
# sudo apt-get install libta-lib-dev && pip install TA-Lib

# Configurasi file .env

# Run Script
python3 binance.py


