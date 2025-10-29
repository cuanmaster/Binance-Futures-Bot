import ccxt
import pandas as pd
import numpy as np
import time
import json
from datetime import datetime
import google.generativeai as genai
import talib
import asyncio
import telegram
from telegram import Bot
from concurrent.futures import ThreadPoolExecutor
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class BinanceFuturesBot:
    def __init__(self):
        """
        Inisialisasi bot trading dengan environment variables
        """
        # Load configuration from environment
        self.api_key = os.getenv('BINANCE_API_KEY')
        self.api_secret = os.getenv('BINANCE_API_SECRET')
        self.gemini_api_key = os.getenv('GEMINI_API_KEY')
        self.telegram_token = os.getenv('TELEGRAM_BOT_TOKEN')
        self.telegram_chat_id = os.getenv('TELEGRAM_CHAT_ID')
        
        # Validate required environment variables
        self._validate_environment()
        
        # Setup Binance
        self.exchange = ccxt.binance({
            'apiKey': self.api_key,
            'secret': self.api_secret,
            'enableRateLimit': True,
            'options': {
                'defaultType': 'future',
            }
        })
        
        # Setup Gemini AI
        genai.configure(api_key=self.gemini_api_key)
        self.gemini_model = genai.GenerativeModel('gemini-1.5-flash')
        
        # Setup Telegram Bot
        self.telegram_bot = Bot(token=self.telegram_token)
        self.telegram_chat_id = self.telegram_chat_id
        
        # Multi-pair symbols from environment
        symbols_env = os.getenv('TRADING_SYMBOLS', 'BTC/USDT,ETH/USDT,BNB/USDT,SOL/USDT,XRP/USDT')
        self.symbols = [symbol.strip() for symbol in symbols_env.split(',')]
        
        # Trading parameters from environment
        self.timeframe = os.getenv('TIMEFRAME', '15m')
        self.leverage = int(os.getenv('TRADING_LEVERAGE', '10'))
        self.risk_percent = float(os.getenv('RISK_PERCENT', '2'))
        self.take_profit_percent = float(os.getenv('TAKE_PROFIT_PERCENT', '3'))
        self.stop_loss_percent = float(os.getenv('STOP_LOSS_PERCENT', '1.5'))
        self.max_positions = int(os.getenv('MAX_POSITIONS', '3'))
        
        # AI optimization parameters from environment
        self.ai_threshold = int(os.getenv('AI_THRESHOLD', '3'))
        self.ai_min_confidence = float(os.getenv('AI_MIN_CONFIDENCE', '70'))
        
        # Initialize tracking variables
        self.positions = {}
        self.orders = {}
        self.total_trades = 0
        self.winning_trades = 0
        self.total_profit = 0
        self.ai_requests = 0
        self.ai_requests_saved = 0
        
        print(f"✅ Bot initialized for {len(self.symbols)} pairs")
        self.send_telegram_sync(f"🤖 Bot Started!\n\nSymbols: {', '.join(self.symbols)}\nLeverage: {self.leverage}x\nMax Positions: {self.max_positions}")
        self.set_leverage_all()
    
    def _validate_environment(self):
        """Validate that all required environment variables are set"""
        required_vars = [
            'BINANCE_API_KEY',
            'BINANCE_API_SECRET', 
            'GEMINI_API_KEY',
            'TELEGRAM_BOT_TOKEN',
            'TELEGRAM_CHAT_ID'
        ]
        
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        if missing_vars:
            raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")
    
    def send_telegram_sync(self, message):
        """Send telegram notification synchronously"""
        try:
            asyncio.run(self.send_telegram(message))
        except Exception as e:
            print(f"⚠️ Telegram error: {e}")
    
    async def send_telegram(self, message):
        """Send notification via Telegram"""
        try:
            await self.telegram_bot.send_message(
                chat_id=self.telegram_chat_id,
                text=message,
                parse_mode='HTML'
            )
        except Exception as e:
            print(f"⚠️ Telegram send error: {e}")
    
    def set_leverage_all(self):
        """Set leverage untuk semua symbols"""
        for symbol in self.symbols:
            try:
                # Method yang benar untuk set leverage di Binance Futures
                self.exchange.set_leverage(
                    leverage=self.leverage,
                    symbol=symbol.replace('/', '')
                )
                print(f"✅ Leverage set to {self.leverage}x for {symbol}")
            except Exception as e:
                print(f"⚠️ Leverage setting error for {symbol}: {e}")
    
    def fetch_ohlcv(self, symbol, limit=100):
        """Fetch candlestick data"""
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, self.timeframe, limit=limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            return df
        except Exception as e:
            print(f"❌ Error fetching data for {symbol}: {e}")
            return None
    
    def calculate_indicators(self, df):
        """Hitung indikator teknikal"""
        try:
            # RSI
            df['rsi'] = talib.RSI(df['close'], timeperiod=14)
            
            # MACD
            df['macd'], df['macd_signal'], df['macd_hist'] = talib.MACD(
                df['close'], fastperiod=12, slowperiod=26, signalperiod=9
            )
            
            # Bollinger Bands
            df['bb_upper'], df['bb_middle'], df['bb_lower'] = talib.BBANDS(
                df['close'], timeperiod=20, nbdevup=2, nbdevdn=2
            )
            
            # EMA
            df['ema_9'] = talib.EMA(df['close'], timeperiod=9)
            df['ema_21'] = talib.EMA(df['close'], timeperiod=21)
            df['ema_50'] = talib.EMA(df['close'], timeperiod=50)
            
            # ATR untuk volatilitas
            df['atr'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)
            
            # Volume analysis
            df['volume_sma'] = talib.SMA(df['volume'], timeperiod=20)
            
            # Stochastic
            df['stoch_k'], df['stoch_d'] = talib.STOCH(df['high'], df['low'], df['close'])
            
            return df
        except Exception as e:
            print(f"❌ Error calculating indicators: {e}")
            return df
    
    def analyze_with_gemini(self, symbol, df):
        """Analisis market menggunakan Gemini AI"""
        try:
            self.ai_requests += 1
            
            latest = df.iloc[-1]
            prev = df.iloc[-2]
            
            # Prepare data untuk Gemini
            analysis_prompt = f"""
            Analisis data trading untuk {symbol}:
            
            Harga: ${latest['close']:.2f}
            RSI: {latest['rsi']:.2f}
            MACD: {latest['macd']:.4f} | Signal: {latest['macd_signal']:.4f}
            Stochastic K: {latest['stoch_k']:.2f}
            
            Bollinger Bands:
            - Upper: ${latest['bb_upper']:.2f}
            - Middle: ${latest['bb_middle']:.2f}
            - Lower: ${latest['bb_lower']:.2f}
            
            EMA: 9=${latest['ema_9']:.2f} | 21=${latest['ema_21']:.2f} | 50=${latest['ema_50']:.2f}
            
            Volume: {latest['volume']:.0f} vs Avg: {latest['volume_sma']:.0f}
            
            Berikan rekomendasi: LONG, SHORT, atau HOLD
            Format: [SIGNAL]|[CONFIDENCE%]|[REASONING]
            """
            
            response = self.gemini_model.generate_content(analysis_prompt)
            analysis = response.text.strip()
            
            # Parse response
            parts = analysis.split('|')
            if len(parts) >= 2:
                signal = parts[0].strip().upper()
                confidence = parts[1].strip().replace('%', '')
                reasoning = parts[2].strip() if len(parts) > 2 else "No reason provided"
                
                return {
                    'signal': signal,
                    'confidence': float(confidence) if confidence.replace('.','').isdigit() else 50,
                    'reasoning': reasoning
                }
            
            return {'signal': 'HOLD', 'confidence': 0, 'reasoning': 'Unable to parse AI response'}
            
        except Exception as e:
            print(f"⚠️ Gemini AI error for {symbol}: {e}")
            return {'signal': 'HOLD', 'confidence': 0, 'reasoning': str(e)}
    
    def get_technical_signal(self, df):
        """Analisis teknikal manual"""
        try:
            latest = df.iloc[-1]
            prev = df.iloc[-2]
            
            signals = []
            score = 0  # Bullish: positive, Bearish: negative
            
            # RSI signals (check for NaN)
            if not pd.isna(latest['rsi']):
                if latest['rsi'] < 30:
                    signals.append(('LONG', 'RSI oversold'))
                    score += 2
                elif latest['rsi'] > 70:
                    signals.append(('SHORT', 'RSI overbought'))
                    score -= 2
            
            # MACD crossover (check for NaN)
            if not pd.isna(latest['macd']) and not pd.isna(prev['macd']):
                if prev['macd'] < prev['macd_signal'] and latest['macd'] > latest['macd_signal']:
                    signals.append(('LONG', 'MACD bullish crossover'))
                    score += 2
                elif prev['macd'] > prev['macd_signal'] and latest['macd'] < latest['macd_signal']:
                    signals.append(('SHORT', 'MACD bearish crossover'))
                    score -= 2
            
            # EMA trend (check for NaN)
            if not pd.isna(latest['ema_9']) and not pd.isna(latest['ema_21']) and not pd.isna(latest['ema_50']):
                if latest['ema_9'] > latest['ema_21'] > latest['ema_50']:
                    signals.append(('LONG', 'EMA bullish alignment'))
                    score += 1
                elif latest['ema_9'] < latest['ema_21'] < latest['ema_50']:
                    signals.append(('SHORT', 'EMA bearish alignment'))
                    score -= 1
            
            # Bollinger Bands (check for NaN)
            if not pd.isna(latest['bb_lower']) and not pd.isna(latest['bb_upper']):
                if latest['close'] < latest['bb_lower']:
                    signals.append(('LONG', 'Price below BB lower'))
                    score += 1
                elif latest['close'] > latest['bb_upper']:
                    signals.append(('SHORT', 'Price above BB upper'))
                    score -= 1
            
            # Stochastic (check for NaN)
            if not pd.isna(latest['stoch_k']):
                if latest['stoch_k'] < 20:
                    signals.append(('LONG', 'Stochastic oversold'))
                    score += 1
                elif latest['stoch_k'] > 80:
                    signals.append(('SHORT', 'Stochastic overbought'))
                    score -= 1
            
            return signals, score
            
        except Exception as e:
            print(f"⚠️ Error in technical signal calculation: {e}")
            return [], 0
    
    def get_balance(self):
        """Get USDT balance"""
        try:
            balance = self.exchange.fetch_balance()
            return balance['USDT']['free']
        except Exception as e:
            print(f"❌ Error fetching balance: {e}")
            return 0
    
    def calculate_position_size(self, entry_price, stop_loss_price):
        """Hitung ukuran posisi berdasarkan risk management"""
        try:
            balance = self.get_balance()
            
            # Adjust risk per active positions
            active_positions = len([p for p in self.positions.values() if p is not None])
            adjusted_risk = self.risk_percent / max(1, active_positions)
            
            risk_amount = balance * (adjusted_risk / 100)
            
            price_diff = abs(entry_price - stop_loss_price)
            position_size = (risk_amount / price_diff) * entry_price
            
            # Limit to available balance
            max_position = balance * self.leverage * 0.95  # 95% of max
            position_size = min(position_size, max_position / len(self.symbols))
            
            return round(position_size / entry_price, 3)
        except Exception as e:
            print(f"❌ Error calculating position size: {e}")
            return 0
    
    def open_position(self, symbol, side, entry_price, df, reasoning):
        """Buka posisi trading"""
        try:
            # Check max positions limit
            active_positions = len([p for p in self.positions.values() if p is not None])
            if active_positions >= self.max_positions:
                print(f"⚠️ Max positions ({self.max_positions}) reached")
                return False
            
            latest = df.iloc[-1]
            
            # Calculate stop loss and take profit
            if side == 'LONG':
                stop_loss = entry_price * (1 - self.stop_loss_percent / 100)
                take_profit = entry_price * (1 + self.take_profit_percent / 100)
            else:  # SHORT
                stop_loss = entry_price * (1 + self.stop_loss_percent / 100)
                take_profit = entry_price * (1 - self.take_profit_percent / 100)
            
            # Calculate position size
            amount = self.calculate_position_size(entry_price, stop_loss)
            
            if amount <= 0:
                print(f"❌ Invalid position size for {symbol}")
                return False
            
            # Place market order
            order_side = 'buy' if side == 'LONG' else 'sell'
            order = self.exchange.create_order(
                symbol=symbol,
                type='market',
                side=order_side,
                amount=amount
            )
            
            print(f"\n✅ {side} Position Opened for {symbol}!")
            print(f"Entry: ${entry_price:.2f}")
            print(f"Amount: {amount}")
            print(f"Stop Loss: ${stop_loss:.2f}")
            print(f"Take Profit: ${take_profit:.2f}")
            
            # Save position info
            self.positions[symbol] = {
                'side': side,
                'entry_price': entry_price,
                'amount': amount,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'order_id': order['id'],
                'timestamp': datetime.now()
            }
            
            # Place stop loss and take profit orders
            self.place_exit_orders(symbol, side, amount, stop_loss, take_profit)
            
            # Send Telegram notification
            message = f"""
🚀 <b>NEW POSITION OPENED</b>

📊 Symbol: <b>{symbol}</b>
📈 Side: <b>{side}</b>
💰 Entry: ${entry_price:.2f}
📦 Amount: {amount}

🎯 Take Profit: ${take_profit:.2f} (+{self.take_profit_percent}%)
🛑 Stop Loss: ${stop_loss:.2f} (-{self.stop_loss_percent}%)

🤖 Reason: {reasoning[:100]}

💼 Active Positions: {active_positions + 1}/{self.max_positions}
            """
            self.send_telegram_sync(message)
            
            self.total_trades += 1
            
            return True
            
        except Exception as e:
            print(f"❌ Error opening position for {symbol}: {e}")
            error_msg = f"❌ Error opening {side} position for {symbol}\n{str(e)}"
            self.send_telegram_sync(error_msg)
            return False
    
    def place_exit_orders(self, symbol, side, amount, stop_loss, take_profit):
        """Place stop loss dan take profit orders"""
        try:
            sl_side = 'sell' if side == 'LONG' else 'buy'
            
            # Stop loss order
            sl_order = self.exchange.create_order(
                symbol=symbol,
                type='stop_market',
                side=sl_side,
                amount=amount,
                params={'stopPrice': stop_loss}
            )
            
            # Take profit order
            tp_order = self.exchange.create_order(
                symbol=symbol,
                type='take_profit_market',
                side=sl_side,
                amount=amount,
                params={'stopPrice': take_profit}
            )
            
            self.orders[symbol] = [sl_order['id'], tp_order['id']]
            print(f"✅ Exit orders placed for {symbol}")
            
            # Send notification untuk TP/SL orders
            orders_msg = f"""
📋 <b>EXIT ORDERS PLACED</b>

📊 Symbol: <b>{symbol}</b>

🎯 <b>Take Profit Order</b>
Price: ${take_profit:.2f}
Type: Take Profit Market
Order ID: {tp_order['id']}

🛑 <b>Stop Loss Order</b>
Price: ${stop_loss:.2f}
Type: Stop Market
Order ID: {sl_order['id']}

⚡ Orders akan execute otomatis saat price tercapai
            """
            self.send_telegram_sync(orders_msg)
            
        except Exception as e:
            print(f"⚠️ Error placing exit orders for {symbol}: {e}")
            error_msg = f"⚠️ <b>Failed to place exit orders for {symbol}</b>\n\n{str(e)}\n\n⚠️ MANUAL MONITORING REQUIRED!"
            self.send_telegram_sync(error_msg)
    
    def check_position_status(self, symbol):
        """Check apakah posisi masih aktif dan deteksi TP/SL"""
        try:
            positions = self.exchange.fetch_positions([symbol.replace('/', '')])
            
            for pos in positions:
                if float(pos['contracts']) > 0:
                    # Posisi masih aktif, cek price vs TP/SL
                    if symbol in self.positions and self.positions[symbol] is not None:
                        current_price = float(pos['markPrice'])
                        position_data = self.positions[symbol]
                        
                        # Deteksi apakah mendekati TP atau SL
                        if position_data['side'] == 'LONG':
                            tp_distance = ((position_data['take_profit'] - current_price) / current_price) * 100
                            sl_distance = ((current_price - position_data['stop_loss']) / current_price) * 100
                        else:  # SHORT
                            tp_distance = ((current_price - position_data['take_profit']) / current_price) * 100
                            sl_distance = ((position_data['stop_loss'] - current_price) / current_price) * 100
                        
                        # Alert jika mendekati TP (dalam 0.5%)
                        if tp_distance < 0.5 and not position_data.get('tp_alert_sent', False):
                            alert_msg = f"""
⚠️ <b>APPROACHING TAKE PROFIT</b>

📊 Symbol: <b>{symbol}</b>
📈 Side: <b>{position_data['side']}</b>
💰 Current: ${current_price:.2f}
🎯 Take Profit: ${position_data['take_profit']:.2f}
📏 Distance: {tp_distance:.2f}%
                            """
                            self.send_telegram_sync(alert_msg)
                            position_data['tp_alert_sent'] = True
                        
                        # Alert jika mendekati SL (dalam 0.5%)
                        if sl_distance < 0.5 and not position_data.get('sl_alert_sent', False):
                            alert_msg = f"""
🚨 <b>APPROACHING STOP LOSS</b>

📊 Symbol: <b>{symbol}</b>
📈 Side: <b>{position_data['side']}</b>
💰 Current: ${current_price:.2f}
🛑 Stop Loss: ${position_data['stop_loss']:.2f}
📏 Distance: {sl_distance:.2f}%
                            """
                            self.send_telegram_sync(alert_msg)
                            position_data['sl_alert_sent'] = True
                    
                    return True
            
            # Jika tidak ada posisi aktif, berarti sudah closed
            if symbol in self.positions and self.positions[symbol] is not None:
                self.handle_position_close(symbol)
            
            return False
            
        except Exception as e:
            print(f"⚠️ Error checking position for {symbol}: {e}")
            return False
    
    def handle_position_close(self, symbol):
        """Handle saat posisi closed dan deteksi apakah TP atau SL"""
        try:
            position_data = self.positions[symbol]
            current_price = self.exchange.fetch_ticker(symbol)['last']
            profit = self.calculate_profit(symbol)
            
            # Deteksi apakah closed karena TP atau SL
            close_reason = "MANUAL/OTHER"
            
            if position_data['side'] == 'LONG':
                # Cek jika price mendekati TP
                if abs(current_price - position_data['take_profit']) / current_price < 0.01:
                    close_reason = "TAKE PROFIT"
                # Cek jika price mendekati SL
                elif abs(current_price - position_data['stop_loss']) / current_price < 0.01:
                    close_reason = "STOP LOSS"
            else:  # SHORT
                if abs(current_price - position_data['take_profit']) / current_price < 0.01:
                    close_reason = "TAKE PROFIT"
                elif abs(current_price - position_data['stop_loss']) / current_price < 0.01:
                    close_reason = "STOP LOSS"
            
            print(f"\n✅ Position Closed for {symbol}!")
            print(f"Close Reason: {close_reason}")
            print(f"Profit/Loss: ${profit:.2f}")
            
            # Update statistics
            self.total_profit += profit
            if profit > 0:
                self.winning_trades += 1
            
            win_rate = (self.winning_trades / self.total_trades * 100) if self.total_trades > 0 else 0
            
            # Emoji based on close reason and profit
            if close_reason == "TAKE PROFIT":
                result_emoji = "🎯✅"
            elif close_reason == "STOP LOSS":
                result_emoji = "🛑❌"
            else:
                result_emoji = "✅" if profit > 0 else "❌"
            
            # Calculate profit percentage
            profit_pct = (profit / (position_data['entry_price'] * position_data['amount'])) * 100
            
            # Detailed close message
            message = f"""
{result_emoji} <b>POSITION CLOSED - {close_reason}</b>

📊 Symbol: <b>{symbol}</b>
📈 Side: <b>{position_data['side']}</b>
💰 Entry: ${position_data['entry_price']:.2f}
💵 Exit: ${current_price:.2f}
📦 Amount: {position_data['amount']}

💸 <b>Profit/Loss: ${profit:.2f} ({profit_pct:+.2f}%)</b>

🎯 TP was: ${position_data['take_profit']:.2f}
🛑 SL was: ${position_data['stop_loss']:.2f}

📊 <b>Overall Statistics</b>
Total Trades: {self.total_trades}
Winning Trades: {self.winning_trades}
Win Rate: {win_rate:.1f}%
Total Profit: ${self.total_profit:.2f}

🕐 Closed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            """
            self.send_telegram_sync(message)
            
            # Reset position
            self.positions[symbol] = None
            if symbol in self.orders:
                del self.orders[symbol]
            
        except Exception as e:
            print(f"⚠️ Error handling position close for {symbol}: {e}")
    
    def calculate_profit(self, symbol):
        """Hitung profit/loss"""
        try:
            # Check if position exists
            if symbol not in self.positions or self.positions[symbol] is None:
                return 0
            
            position = self.positions[symbol]
            current_price = self.exchange.fetch_ticker(symbol)['last']
            entry = position['entry_price']
            amount = position['amount']
            
            if position['side'] == 'LONG':
                profit = (current_price - entry) * amount
            else:
                profit = (entry - current_price) * amount
            
            return profit
            
        except Exception as e:
            print(f"⚠️ Error calculating profit for {symbol}: {e}")
            return 0
    
    def analyze_symbol(self, symbol):
        """Analyze single symbol"""
        try:
            # Initialize position if not exists
            if symbol not in self.positions:
                self.positions[symbol] = None
            
            # Check if we already have position for this symbol
            has_position = self.check_position_status(symbol)
            
            if has_position and symbol in self.positions and self.positions[symbol] is not None:
                # Monitor existing position
                current_price = self.exchange.fetch_ticker(symbol)['last']
                profit = self.calculate_profit(symbol)
                position = self.positions[symbol]
                
                # Safe division check
                position_value = position['entry_price'] * position['amount']
                profit_pct = (profit / position_value * 100) if position_value > 0 else 0
                
                # Tampilkan monitoring info
                print(f"📊 {symbol} | {position['side']} | Entry: ${position['entry_price']:.2f} | Current: ${current_price:.2f} | P/L: ${profit:.2f} ({profit_pct:+.2f}%)")
                
                # Kirim update setiap 30 menit untuk posisi aktif (opsional)
                last_update = position.get('last_update', None)
                if last_update is None or (datetime.now() - last_update).total_seconds() > 1800:
                    status_msg = f"""
📊 <b>POSITION UPDATE</b>

Symbol: <b>{symbol}</b>
Side: <b>{position['side']}</b>
Entry: ${position['entry_price']:.2f}
Current: ${current_price:.2f}

Current P/L: <b>${profit:.2f} ({profit_pct:+.2f}%)</b>

🎯 TP: ${position['take_profit']:.2f}
🛑 SL: ${position['stop_loss']:.2f}

Status: Position Active ✅
                    """
                    # Uncomment jika ingin update rutin
                    # self.send_telegram_sync(status_msg)
                    self.positions[symbol]['last_update'] = datetime.now()
                
                return
            
            # Fetch and analyze data
            df = self.fetch_ohlcv(symbol)
            if df is None or len(df) < 50:
                print(f"⚠️ Insufficient data for {symbol}")
                return
            
            # Calculate indicators
            df = self.calculate_indicators(df)
            
            # Check if indicators calculated properly
            if df['rsi'].isna().all() or df['macd'].isna().all():
                print(f"⚠️ Invalid indicators for {symbol}")
                return
            
            # Get technical signals first
            tech_signals, tech_score = self.get_technical_signal(df)
            
            # Only call AI if technical score meets threshold
            if abs(tech_score) >= self.ai_threshold:
                ai_analysis = self.analyze_with_gemini(symbol, df)
                print(f"🔍 {symbol} | Tech Score: {tech_score} | AI: {ai_analysis['signal']} ({ai_analysis['confidence']}%)")
                
                # Decision making with AI
                if ai_analysis['confidence'] >= self.ai_min_confidence and abs(tech_score) >= 2:
                    signal = ai_analysis['signal']
                    
                    # Confirm AI and technical signals align
                    if (signal == 'LONG' and tech_score > 0) or (signal == 'SHORT' and tech_score < 0):
                        current_price = df.iloc[-1]['close']
                        
                        print(f"🎯 Trading Signal for {symbol}: {signal}")
                        print(f"Confidence: {ai_analysis['confidence']}%")
                        
                        # Open position
                        self.open_position(symbol, signal, current_price, df, ai_analysis['reasoning'])
            else:
                print(f"🔍 {symbol} | Tech Score: {tech_score} | Skipping AI (below threshold)")
                self.ai_requests_saved += 1
            
        except KeyError as ke:
            print(f"❌ KeyError analyzing {symbol}: {ke} - Possibly missing data or position info")
        except Exception as e:
            print(f"❌ Error analyzing {symbol}: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def send_daily_report(self):
        """Send daily performance report"""
        try:
            active_positions = len([p for p in self.positions.values() if p is not None])
            win_rate = (self.winning_trades / self.total_trades * 100) if self.total_trades > 0 else 0
            
            message = f"""
📊 <b>DAILY REPORT</b>

💼 Active Positions: {active_positions}/{self.max_positions}
📈 Total Trades: {self.total_trades}
✅ Winning Trades: {self.winning_trades}
📊 Win Rate: {win_rate:.1f}%
💰 Total Profit: ${self.total_profit:.2f}

🤖 AI Statistics:
- AI Requests: {self.ai_requests}
- AI Requests Saved: {self.ai_requests_saved}

🕐 Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            """
            
            self.send_telegram_sync(message)
            
        except Exception as e:
            print(f"⚠️ Error sending daily report: {e}")
    
    def run(self):
        """Main trading loop"""
        print("\n🚀 Starting Multi-Pair Binance Futures Trading Bot...")
        print(f"Symbols: {', '.join(self.symbols)}")
        print(f"Timeframe: {self.timeframe}")
        print(f"Leverage: {self.leverage}x")
        print(f"Max Positions: {self.max_positions}")
        print(f"AI Threshold: {self.ai_threshold}")
        print(f"AI Min Confidence: {self.ai_min_confidence}%")
        print("-" * 50)
        
        last_report_time = datetime.now()
        
        while True:
            try:
                print(f"\n{'='*60}")
                print(f"🔄 Scanning all symbols... {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"{'='*60}")
                
                # Analyze all symbols
                for symbol in self.symbols:
                    self.analyze_symbol(symbol)
                    time.sleep(2)  # Delay to avoid rate limits
                
                # Send daily report every 24 hours
                if (datetime.now() - last_report_time).total_seconds() > 86400:
                    self.send_daily_report()
                    last_report_time = datetime.now()
                
                # Sleep for next iteration
                sleep_time = 60
                print(f"\n💤 Sleeping for {sleep_time}s...")
                time.sleep(sleep_time)
                
            except KeyboardInterrupt:
                print("\n\n⛔ Bot stopped by user")
                self.send_telegram_sync("⛔ Bot stopped by user")
                break
            except Exception as e:
                print(f"\n❌ Error in main loop: {e}")
                time.sleep(60)

# ============= USAGE =============
if __name__ == "__main__":
    try:
        # Create and run bot dengan environment variables
        bot = BinanceFuturesBot()
        
        # Start trading
        bot.run()
        
    except ValueError as e:
        print(f"❌ Configuration Error: {e}")
        print("Please check your .env file and ensure all required variables are set.")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
