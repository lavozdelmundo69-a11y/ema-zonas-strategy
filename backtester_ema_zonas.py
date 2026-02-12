#!/usr/bin/env python3
"""
Backtester EMA 3/22 + Zonas Mitigadas (Order Blocks + Liquidez)
Soporta intervalos intradía (30m, 15m, etc.) con hasta 60 días de datos.
Busca >5% anualizado.
"""

import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

class EstrategiaEMAZonas:
    def __init__(self, fast_ema=3, slow_ema=22, bars_window=10,
                 swing_len=16, liq_tolerance=0.5,
                 ob_lookback=5, ob_min_size=2.0, use_volume=True,
                 atr_len=14, tp_mult=3.0, sl_mult=2.0,
                 use_liq=True, use_ob=True):
        self.fast_ema = fast_ema
        self.slow_ema = slow_ema
        self.bars_window = bars_window
        self.swing_len = swing_len
        self.liq_tolerance = liq_tolerance / 100.0
        self.ob_lookback = ob_lookback
        self.ob_min_size = ob_min_size / 100.0
        self.use_volume = use_volume
        self.atr_len = atr_len
        self.tp_mult = tp_mult
        self.sl_mult = sl_mult
        self.use_liq = use_liq
        self.use_ob = use_ob
        
        self.liq_highs = []
        self.liq_lows = []
        self.ob_zones = []
        
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df['ema_fast'] = df['close'].ewm(span=self.fast_ema, adjust=False).mean()
        df['ema_slow'] = df['close'].ewm(span=self.slow_ema, adjust=False).mean()
        
        # ATR
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        df['atr'] = true_range.rolling(window=self.atr_len).mean()
        
        if self.use_volume:
            df['volume_sma'] = df['volume'].rolling(20).mean()
        
        return df
    
    def check_liquidity_mitigation(self, close: float, bar_idx: int):
        for arr in [self.liq_highs, self.liq_lows]:
            for zone in arr:
                if zone['mitigated']:
                    continue
                price = zone['price']
                if arr is self.liq_highs:
                    mitigated = close > price * (1 + self.liq_tolerance)
                else:
                    mitigated = close < price * (1 - self.liq_tolerance)
                if mitigated:
                    zone['mitigated'] = True
                    zone['mitig_bar'] = bar_idx

    def check_ob_mitigation(self, close: float, bar_idx: int):
        for zone in self.ob_zones:
            if zone['mitigated']:
                continue
            h, l, is_bull = zone['high'], zone['low'], zone['is_bull']
            if is_bull:
                mitigated = close <= h
            else:
                mitigated = close >= l
            if mitigated:
                zone['mitigated'] = True
                zone['mitig_bar'] = bar_idx

    def recent_zone_mitigated(self, bar_idx: int) -> bool:
        window_start = bar_idx - self.bars_window
        all_zones = self.liq_highs + self.liq_lows + self.ob_zones
        for zone in all_zones:
            if zone['mitigated'] and zone['mitig_bar'] >= window_start:
                return True
        return False

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        n = len(df)
        df['signal'] = 0
        df['zone_mitigated'] = False
        
        self.liq_highs.clear()
        self.liq_lows.clear()
        self.ob_zones.clear()
        
        # Pre-calcular pivotes confirmables
        pending_highs = []
        pending_lows = []
        for i in range(self.swing_len, n):
            p = i - self.swing_len
            if p < self.swing_len:
                continue
            left_max = df['high'].iloc[p - self.swing_len : p].max()
            right_max = df['high'].iloc[p+1 : i+1].max()
            if df['high'].iloc[p] > left_max and df['high'].iloc[p] > right_max:
                pending_highs.append((i, p, df['high'].iloc[p]))
            left_min = df['low'].iloc[p - self.swing_len : p].min()
            right_min = df['low'].iloc[p+1 : i+1].min()
            if df['low'].iloc[p] < left_min and df['low'].iloc[p] < right_min:
                pending_lows.append((i, p, df['low'].iloc[p]))
                
        pending_highs.sort(key=lambda x: x[0])
        pending_lows.sort(key=lambda x: x[0])
        idx_high = 0
        idx_low = 0
        
        for i in range(n):
            # Añadir zonas de liquidez confirmadas
            while idx_high < len(pending_highs) and pending_highs[idx_high][0] == i:
                conf, origin, price = pending_highs[idx_high]
                self.liq_highs.append({'price': price, 'bar_idx': origin, 'mitigated': False, 'mitig_bar': -1})
                idx_high += 1
            while idx_low < len(pending_lows) and pending_lows[idx_low][0] == i:
                conf, origin, price = pending_lows[idx_low]
                self.liq_lows.append({'price': price, 'bar_idx': origin, 'mitigated': False, 'mitig_bar': -1})
                idx_low += 1
            
            # Detectar Order Blocks
            if i >= 1 and self.use_ob:
                look = min(self.ob_lookback, i)
                price_i = df.iloc[i]['close']
                price_look = df.iloc[i-look]['close']
                # Bull OB
                momentum_up = (price_i - price_look) / price_look
                prev_bearish = df.iloc[i-1]['open'] > df.iloc[i-1]['close']
                strong_up = momentum_up >= self.ob_min_size
                if prev_bearish and strong_up and price_i > df.iloc[i]['open']:
                    ok = True
                    if self.use_volume and 'volume_sma' in df.columns:
                        avg_vol = df.iloc[i-1]['volume_sma']
                        ok = df.iloc[i-1]['volume'] > avg_vol * 1.2
                    if ok:
                        self.ob_zones.append({
                            'high': df.iloc[i-1]['high'],
                            'low': df.iloc[i-1]['low'],
                            'bar_idx': i-1,
                            'is_bull': True,
                            'mitigated': False,
                            'mitig_bar': -1
                        })
                # Bear OB
                momentum_down = (price_look - price_i) / price_i
                prev_bullish = df.iloc[i-1]['open'] < df.iloc[i-1]['close']
                strong_down = momentum_down >= self.ob_min_size
                if prev_bullish and strong_down and price_i < df.iloc[i]['open']:
                    ok = True
                    if self.use_volume and 'volume_sma' in df.columns:
                        avg_vol = df.iloc[i-1]['volume_sma']
                        ok = df.iloc[i-1]['volume'] > avg_vol * 1.2
                    if ok:
                        self.ob_zones.append({
                            'high': df.iloc[i-1]['high'],
                            'low': df.iloc[i-1]['low'],
                            'bar_idx': i-1,
                            'is_bull': False,
                            'mitigated': False,
                            'mitig_bar': -1
                        })
            
            # Mitigaciones
            close_i = df.iloc[i]['close']
            self.check_liquidity_mitigation(close_i, i)
            self.check_ob_mitigation(close_i, i)
            
            # EMA cross
            if i < 1:
                continue
            prev_fast = df.iloc[i-1]['ema_fast']
            prev_slow = df.iloc[i-1]['ema_slow']
            curr_fast = df.iloc[i]['ema_fast']
            curr_slow = df.iloc[i]['ema_slow']
            if pd.isna(prev_fast) or pd.isna(prev_slow) or pd.isna(curr_fast) or pd.isna(curr_slow):
                continue
            golden = (prev_fast <= prev_slow) and (curr_fast > curr_slow)
            dead = (prev_fast >= prev_slow) and (curr_fast < curr_slow)
            zone_ok = self.recent_zone_mitigated(i)
            df.at[df.index[i], 'zone_mitigated'] = zone_ok
            if golden and zone_ok:
                df.at[df.index[i], 'signal'] = 1
            elif dead and zone_ok:
                df.at[df.index[i], 'signal'] = -1
        return df

    def run_backtest(self, df: pd.DataFrame, initial_capital=10000) -> Dict:
        df = self.calculate_indicators(df.copy())
        df = self.generate_signals(df)
        
        capital = initial_capital
        position = 0
        entry_price = 0.0
        entry_bar = 0
        trades = []
        
        for i in range(len(df)):
            signal = df.iloc[i]['signal']
            close = df.iloc[i]['close']
            atr = df.iloc[i]['atr']
            tp_price = close + self.tp_mult * atr
            sl_price = close - self.sl_mult * atr
            
            if position == 0 and signal != 0:
                position = signal
                entry_price = close
                entry_bar = i
            elif position != 0:
                if position == 1 and close <= sl_price:
                    exit_price = sl_price
                    pnl = (exit_price - entry_price) / entry_price * capital
                    capital += pnl
                    trades.append({'entry_bar': entry_bar, 'exit_bar': i, 'type': 'long', 'entry': entry_price, 'exit': exit_price, 'pnl': pnl, 'reason': 'sl', 'exit_date': df.index[i]})
                    position = 0
                elif position == -1 and close >= sl_price:
                    exit_price = sl_price
                    pnl = (entry_price - exit_price) / entry_price * capital
                    capital += pnl
                    trades.append({'entry_bar': entry_bar, 'exit_bar': i, 'type': 'short', 'entry': entry_price, 'exit': exit_price, 'pnl': pnl, 'reason': 'sl', 'exit_date': df.index[i]})
                    position = 0
                elif position == 1 and close >= tp_price:
                    exit_price = tp_price
                    pnl = (exit_price - entry_price) / entry_price * capital
                    capital += pnl
                    trades.append({'entry_bar': entry_bar, 'exit_bar': i, 'type': 'long', 'entry': entry_price, 'exit': exit_price, 'pnl': pnl, 'reason': 'tp', 'exit_date': df.index[i]})
                    position = 0
                elif position == -1 and close <= tp_price:
                    exit_price = tp_price
                    pnl = (entry_price - exit_price) / entry_price * capital
                    capital += pnl
                    trades.append({'entry_bar': entry_bar, 'exit_bar': i, 'type': 'short', 'entry': entry_price, 'exit': exit_price, 'pnl': pnl, 'reason': 'tp', 'exit_date': df.index[i]})
                    position = 0
                    
        if position != 0:
            exit_price = df.iloc[-1]['close']
            if position == 1:
                pnl = (exit_price - entry_price) / entry_price * capital
            else:
                pnl = (entry_price - exit_price) / entry_price * capital
            capital += pnl
            trades.append({'entry_bar': entry_bar, 'exit_bar': len(df)-1, 'type': 'long' if position==1 else 'short', 'entry': entry_price, 'exit': exit_price, 'pnl': pnl, 'reason': 'eof', 'exit_date': df.index[-1]})
            
        return {
            'final_capital': capital,
            'total_return_pct': (capital / initial_capital - 1) * 100,
            'trades': trades,
            'df': df,
            'initial_capital': initial_capital,
            'period_days': (df.index[-1] - df.index[0]).days or 1
        }
    
    def print_stats(self, results: Dict):
        trades = results['trades']
        if not trades:
            print("No se generaron trades en el periodo.")
            return
            
        total_trades = len(trades)
        winning = sum(1 for t in trades if t['pnl'] > 0)
        losing = sum(1 for t in trades if t['pnl'] < 0)
        win_rate = winning / total_trades * 100
        
        total_pnl = sum(t['pnl'] for t in trades)
        avg_win = np.mean([t['pnl'] for t in trades if t['pnl'] > 0]) if winning else 0
        avg_loss = np.mean([t['pnl'] for t in trades if t['pnl'] < 0]) if losing else 0
        gross_profit = sum(t['pnl'] for t in trades if t['pnl'] > 0)
        gross_loss = abs(sum(t['pnl'] for t in trades if t['pnl'] < 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf
        
        # Anualizar retorno
        period_days = results['period_days']
        years = period_days / 365.0
        annual_return = ((1 + results['total_return_pct']/100) ** (1/years) - 1) * 100 if years > 0 else results['total_return_pct']
        
        print("\n" + "="*70)
        print("BACKTEST RESULTS — EMA 3/22 + ZONAS OB/LIQ MITIGADAS")
        print("="*70)
        print(f"Periodo:            {results['df'].index[0].date()} → {results['df'].index[-1].date()} ({period_days} días)")
        print(f"Capital inicial:    ${results['initial_capital']:,.2f}")
        print(f"Capital final:      ${results['final_capital']:,.2f}")
        print(f"Retorno total:      {results['total_return_pct']:.2f}%")
        print(f"Retorno anualizado: {annual_return:.2f}%")
        print(f"Total trades:       {total_trades}")
        print(f"Win rate:           {win_rate:.2f}%")
        print(f"Ganadores:         {winning}")
        print(f"Perdedores:        {losing}")
        print(f"Gross Profit:      ${gross_profit:,.2f}")
        print(f"Gross Loss:        ${gross_loss:,.2f}")
        print(f"Profit Factor:     {profit_factor:.2f}")
        print(f"Avg win:           ${avg_win:,.2f}")
        print(f"Avg loss:          ${avg_loss:,.2f}")
        
        longs = sum(1 for t in trades if t['type']=='long')
        shorts = sum(1 for t in trades if t['type']=='short')
        print(f"Longs: {longs} | Shorts: {shorts}")
        
        reasons = {}
        for t in trades:
            r = t['reason']
            reasons[r] = reasons.get(r, 0) + 1
        print("\nRazones de salida:")
        for r, c in reasons.items():
            print(f"  {r}: {c} ({c/total_trades*100:.1f}%)")
        print("="*70)
        
        print("\nPrimeros 5 trades:")
        for i, t in enumerate(trades[:5]):
            print(f"{i+1}. {t['type'].upper()} | Entry: {t['entry']:.2f} @ {t['entry_bar']} | Exit: {t['exit']:.2f} @ {t['exit_bar']} | PnL: ${t['pnl']:.2f} ({t['reason']})")
            
        return {
            'total_return_pct': results['total_return_pct'],
            'annual_return': annual_return,
            'total_trades': total_trades,
            'win_rate': win_rate,
            'profit_factor': profit_factor
        }


def download_data(symbol: str, years: int = 5, interval: str = '1d') -> pd.DataFrame:
    try:
        import yfinance as yf
    except ImportError:
        print("Error: yfinance no instalado. Instálalo con: pip install yfinance")
        sys.exit(1)
        
    is_intraday = interval in ['1m','2m','5m','15m','30m','60m','90m']
    if is_intraday:
        # yfinance solo permite 60 días para intradía
        period = '60d'
        print(f"Descargando {symbol} intervalo {interval} (últimos 60 días)...")
        df = yf.download(symbol, period=period, interval=interval, progress=False)
    else:
        end = datetime.now()
        start = end - timedelta(days=years*365)
        print(f"Descargando {years} años de {symbol} (intervalo {interval})...")
        df = yf.download(symbol, start=start, end=end, progress=False)
    
    if df.empty:
        print(f"No se encontraron datos para {symbol}")
        sys.exit(1)
        
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
    df.columns = ['open', 'high', 'low', 'close', 'volume']
    df.index.name = 'timestamp'
    
    print(f"Datos descargados: {len(df)} barras")
    print(f"Rango: {df.index[0].date()} → {df.index[-1].date()}")
    return df


def load_csv(filepath: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(filepath, parse_dates=['timestamp'])
        df.set_index('timestamp', inplace=True)
        print(f"Datos cargados desde {filepath}: {len(df)} filas")
        required = ['open', 'high', 'low', 'close']
        for col in required:
            if col not in df.columns:
                raise ValueError(f"Falta columna '{col}' en CSV")
        return df[['open','high','low','close','volume']].copy()
    except Exception as e:
        print(f"Error cargando CSV: {e}")
        print("Se requiere columnas: timestamp,open,high,low,close,volume")
        sys.exit(1)


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Backtester EMA 3/22 + Zonas OB/Liq Mitigadas')
    parser.add_argument('csv', nargs='?', help='Archivo CSV con datos')
    parser.add_argument('--download', metavar='SYMBOL', help='Descargar datos de yfinance')
    parser.add_argument('--years', type=int, default=5)
    parser.add_argument('--interval', default='1d', choices=['1m','2m','5m','15m','30m','60m','90m','1d','1wk','1mo'], help='Intervalo de velas')
    parser.add_argument('--fast', type=int, default=3)
    parser.add_argument('--slow', type=int, default=22)
    parser.add_argument('--window', type=int, default=10)
    parser.add_argument('--ob_lookback', type=int, default=5)
    parser.add_argument('--ob_min_size', type=float, default=2.0)
    parser.add_argument('--swing_len', type=int, default=16)
    
    args = parser.parse_args()
    
    if args.download:
        df = download_data(args.download, years=args.years, interval=args.interval)
        csv_name = f"{args.download.replace('-','_')}_{args.interval}_{args.years}y.csv"
        df.to_csv(csv_name)
        print(f"Datos guardados en {csv_name}")
    elif args.csv:
        df = load_csv(args.csv)
    else:
        print("Error: Debes especificar CSV o --download SYMBOL")
        parser.print_help()
        sys.exit(1)
        
    params = {
        'fast_ema': args.fast,
        'slow_ema': args.slow,
        'bars_window': args.window,
        'swing_len': args.swing_len,
        'liq_tolerance': 0.5,
        'ob_lookback': args.ob_lookback,
        'ob_min_size': args.ob_min_size,
        'use_volume': True,
        'atr_len': 14,
        'tp_mult': 3.0,
        'sl_mult': 2.0,
        'use_liq': True,
        'use_ob': True
    }
    
    print("\n" + "="*70)
    print("CONFIGURACIÓN ESTRATEGIA")
    print("="*70)
    for k, v in params.items():
        print(f"{k}: {v}")
    print("-"*70)
    
    strat = EstrategiaEMAZonas(**params)
    results = strat.run_backtest(df)
    strat.print_stats(results)
    
    save = input("\n¿Guardar trades en CSV? (s/N): ").strip().lower()
    if save == 's':
        trades_df = pd.DataFrame(results['trades'])
        trades_df.to_csv('trades_ema_zonas.csv', index=False)
        print("Trades guardados en trades_ema_zonas.csv")
        signals_df = results['df'][['close','ema_fast','ema_slow','signal','zone_mitigated']].copy()
        signals_df.to_csv('signals_ema_zonas.csv')
        print("Señales guardadas en signals_ema_zonas.csv")
    
    # Devolver métricas clave para evaluación automática
    return results


if __name__ == "__main__":
    main()
