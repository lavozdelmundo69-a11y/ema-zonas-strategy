#!/usr/bin/env python3
"""
Backtester Completo para Estrategia EMA 3/22 + Zonas Mitigadas

Detecta:
- Zonas de liquidez: Swing highs/lows (sin lookahead, confirmación con delay)
- Zonas de interés: FVG (Fair Value Gaps)
- Señal: Cruce EMA + zona mitigada reciente (ventana N barras)
- Salida: TP/SL basados en ATR

Uso:
    python backtester_ema_zonas.py datos.csv
    python backtester_ema_zonas.py --download BTC-USD --years 5
"""

import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

class EstrategiaEMAZonas:
    def __init__(self, fast_ema=3, slow_ema=22, bars_window=10,
                 swing_len=16, liq_tolerance=0.5,
                 sens_fvg=0.5, atr_len=14, tp_mult=3.0, sl_mult=2.0,
                 use_liq=True, use_fvg=True):
        self.fast_ema = fast_ema
        self.slow_ema = slow_ema
        self.bars_window = bars_window
        self.swing_len = swing_len
        self.liq_tolerance = liq_tolerance / 100.0
        self.sens_fvg = sens_fvg / 100.0
        self.atr_len = atr_len
        self.tp_mult = tp_mult
        self.sl_mult = sl_mult
        self.use_liq = use_liq
        self.use_fvg = use_fvg
        
        # Arrays persistentes (simulan var arrays de Pine)
        self.liq_highs = []   # [{'price': float, 'bar_idx': int, 'mitigated': bool, 'mitig_bar': int}, ...]
        self.liq_lows = []
        self.fvg_zones = []   # [{'top': float, 'bot': float, 'origin_bar': int, 'mitigated': bool, 'mitig_bar': int, 'is_bull': bool}, ...]
        
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calcula EMAs y ATR"""
        df['ema_fast'] = df['close'].ewm(span=self.fast_ema, adjust=False).mean()
        df['ema_slow'] = df['close'].ewm(span=self.slow_ema, adjust=False).mean()
        
        # ATR
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        df['atr'] = true_range.rolling(window=self.atr_len).mean()
        
        return df
    
    def check_liquidity_mitigation(self, close: float, bar_idx: int):
        """Actualiza estado de mitigación de zonas de liquidez"""
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

    def check_fvg_mitigation(self, close: float, bar_idx: int):
        """Actualiza estado de mitigación de FVGs"""
        for zone in self.fvg_zones:
            if zone['mitigated']:
                continue
            top, bot, is_bull = zone['top'], zone['bot'], zone['is_bull']
            if is_bull:
                mitigated = close <= top
            else:
                mitigated = close >= bot
            if mitigated:
                zone['mitigated'] = True
                zone['mitig_bar'] = bar_idx

    def recent_zone_mitigated(self, bar_idx: int) -> bool:
        """Retorna True si alguna zona se mitigó en la última ventana de barras"""
        window_start = bar_idx - self.bars_window
        # Liquidez highs
        for zone in self.liq_highs:
            if zone['mitigated'] and zone['mitig_bar'] >= window_start:
                return True
        # Liquidez lows
        for zone in self.liq_lows:
            if zone['mitigated'] and zone['mitig_bar'] >= window_start:
                return True
        # FVGs
        for zone in self.fvg_zones:
            if zone['mitigated'] and zone['mitig_bar'] >= window_start:
                return True
        return False

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Genera señales barra a barra sin lookahead.
        Detectar pivotes: solo confirmar en barra i si el pivot ocurrió en p = i - swing_len
        """
        n = len(df)
        df['signal'] = 0
        df['zone_mitigated'] = False  # Debug
        
        # Preparar arrays vacíos
        self.liq_highs.clear()
        self.liq_lows.clear()
        self.fvg_zones.clear()
        
        # Pre-calcular pivotes confirmables por barra
        pending_highs = []  # (conf_bar, origin_bar, price)
        pending_lows = []
        
        for i in range(self.swing_len, n):
            p = i - self.swing_len
            if p < self.swing_len:
                continue
            
            # Swing High: high[p] > todos en [p-swing_len, p-1] y [p+1, p+swing_len]
            left_max = df['high'].iloc[p - self.swing_len : p].max()
            right_max = df['high'].iloc[p+1 : i+1].max()
            if df['high'].iloc[p] > left_max and df['high'].iloc[p] > right_max:
                pending_highs.append((i, p, df['high'].iloc[p]))
                
            # Swing Low
            left_min = df['low'].iloc[p - self.swing_len : p].min()
            right_min = df['low'].iloc[p+1 : i+1].min()
            if df['low'].iloc[p] < left_min and df['low'].iloc[p] < right_min:
                pending_lows.append((i, p, df['low'].iloc[p]))
                
        pending_highs.sort(key=lambda x: x[0])
        pending_lows.sort(key=lambda x: x[0])
        idx_high = 0
        idx_low = 0
        
        # Loop principal
        for i in range(n):
            # Añadir zonas confirmadas esta barra
            while idx_high < len(pending_highs) and pending_highs[idx_high][0] == i:
                conf, origin, price = pending_highs[idx_high]
                self.liq_highs.append({
                    'price': price,
                    'bar_idx': origin,
                    'mitigated': False,
                    'mitig_bar': -1
                })
                idx_high += 1
            while idx_low < len(pending_lows) and pending_lows[idx_low][0] == i:
                conf, origin, price = pending_lows[idx_low]
                self.liq_lows.append({
                    'price': price,
                    'bar_idx': origin,
                    'mitigated': False,
                    'mitig_bar': -1
                })
                idx_low += 1
            
            # FVG
            if i >= 2:
                # Bull FVG
                if df.iloc[i]['low'] > df.iloc[i-2]['high'] and df.iloc[i-1]['close'] > df.iloc[i-1]['open']:
                    top = df.iloc[i]['low']
                    bot = df.iloc[i-2]['high']
                    size_pct = abs(top - bot) / df.iloc[i]['close'] * 100
                    if size_pct >= self.sens_fvg * 100:
                        self.fvg_zones.append({
                            'top': top,
                            'bot': bot,
                            'origin_bar': i-1,
                            'mitigated': False,
                            'mitig_bar': -1,
                            'is_bull': True
                        })
                # Bear FVG
                if df.iloc[i]['high'] < df.iloc[i-2]['low'] and df.iloc[i-1]['close'] < df.iloc[i-1]['open']:
                    top = df.iloc[i-2]['low']
                    bot = df.iloc[i]['high']
                    size_pct = abs(top - bot) / df.iloc[i]['close'] * 100
                    if size_pct >= self.sens_fvg * 100:
                        self.fvg_zones.append({
                            'top': top,
                            'bot': bot,
                            'origin_bar': i-1,
                            'mitigated': False,
                            'mitig_bar': -1,
                            'is_bull': False
                        })
            
            # Mitigaciones
            close_i = df.iloc[i]['close']
            self.check_liquidity_mitigation(close_i, i)
            self.check_fvg_mitigation(close_i, i)
            
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
        """Ejecuta backtest completo con TP/SL dinámicos"""
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
                # Stop Loss
                if position == 1 and close <= sl_price:
                    exit_price = sl_price
                    pnl = (exit_price - entry_price) / entry_price * capital
                    capital += pnl
                    trades.append({
                        'entry_bar': entry_bar,
                        'exit_bar': i,
                        'type': 'long',
                        'entry': entry_price,
                        'exit': exit_price,
                        'pnl': pnl,
                        'reason': 'sl',
                        'exit_date': df.index[i]
                    })
                    position = 0
                elif position == -1 and close >= sl_price:
                    exit_price = sl_price
                    pnl = (entry_price - exit_price) / entry_price * capital
                    capital += pnl
                    trades.append({
                        'entry_bar': entry_bar,
                        'exit_bar': i,
                        'type': 'short',
                        'entry': entry_price,
                        'exit': exit_price,
                        'pnl': pnl,
                        'reason': 'sl',
                        'exit_date': df.index[i]
                    })
                    position = 0
                # Take Profit
                elif position == 1 and close >= tp_price:
                    exit_price = tp_price
                    pnl = (exit_price - entry_price) / entry_price * capital
                    capital += pnl
                    trades.append({
                        'entry_bar': entry_bar,
                        'exit_bar': i,
                        'type': 'long',
                        'entry': entry_price,
                        'exit': exit_price,
                        'pnl': pnl,
                        'reason': 'tp',
                        'exit_date': df.index[i]
                    })
                    position = 0
                elif position == -1 and close <= tp_price:
                    exit_price = tp_price
                    pnl = (entry_price - exit_price) / entry_price * capital
                    capital += pnl
                    trades.append({
                        'entry_bar': entry_bar,
                        'exit_bar': i,
                        'type': 'short',
                        'entry': entry_price,
                        'exit': exit_price,
                        'pnl': pnl,
                        'reason': 'tp',
                        'exit_date': df.index[i]
                    })
                    position = 0
                    
        # Cerrar posición final
        if position != 0:
            exit_price = df.iloc[-1]['close']
            if position == 1:
                pnl = (exit_price - entry_price) / entry_price * capital
            else:
                pnl = (entry_price - exit_price) / entry_price * capital
            capital += pnl
            trades.append({
                'entry_bar': entry_bar,
                'exit_bar': len(df)-1,
                'type': 'long' if position==1 else 'short',
                'entry': entry_price,
                'exit': exit_price,
                'pnl': pnl,
                'reason': 'eof',
                'exit_date': df.index[-1]
            })
            
        return {
            'final_capital': capital,
            'total_return_pct': (capital / initial_capital - 1) * 100,
            'trades': trades,
            'df': df,
            'initial_capital': initial_capital
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
        profit_factor = abs(sum(t['pnl'] for t in trades if t['pnl'] > 0) /
                           sum(t['pnl'] for t in trades if t['pnl'] < 0)) if losing else np.inf
        
        gross_profit = sum(t['pnl'] for t in trades if t['pnl'] > 0)
        gross_loss = abs(sum(t['pnl'] for t in trades if t['pnl'] < 0))
        
        print("\n" + "="*70)
        print("BACKTEST RESULTS — EMA 3/22 + ZONAS MITIGADAS")
        print("="*70)
        print(f"Periodo:            {results['df'].index[0].date()} → {results['df'].index[-1].date()}")
        print(f"Capital inicial:    ${results['initial_capital']:,.2f}")
        print(f"Capital final:      ${results['final_capital']:,.2f}")
        print(f"Retorno total:      {results['total_return_pct']:.2f}%")
        print(f"Total trades:       {total_trades}")
        print(f"Win rate:           {win_rate:.2f}%")
        print(f"Ganadores:         {winning}")
        print(f"Perdedores:        {losing}")
        print(f"Gross Profit:      ${gross_profit:,.2f}")
        print(f"Gross Loss:        ${gross_loss:,.2f}")
        print(f"Profit Factor:     {profit_factor:.2f}")
        print(f"Avg win:           ${avg_win:,.2f}")
        print(f"Avg loss:          ${avg_loss:,.2f}")
        
        # Por tipo
        longs = sum(1 for t in trades if t['type']=='long')
        shorts = sum(1 for t in trades if t['type']=='short')
        print(f"Longs: {longs} | Shorts: {shorts}")
        
        # Razones de salida
        reasons = {}
        for t in trades:
            r = t['reason']
            reasons[r] = reasons.get(r, 0) + 1
        print("\nRazones de salida:")
        for r, c in reasons.items():
            print(f"  {r}: {c} ({c/total_trades*100:.1f}%)")
            
        print("="*70)
        
        # Mostrar algunos trades como ejemplo
        print("\nPrimeros 5 trades:")
        for i, t in enumerate(trades[:5]):
            print(f"{i+1}. {t['type'].upper()} | Entry: {t['entry']:.2f} @ {t['entry_bar']} | "
                  f"Exit: {t['exit']:.2f} @ {t['exit_bar']} | PnL: ${t['pnl']:.2f} ({t['reason']})")


def download_data(symbol: str, years: int = 5) -> pd.DataFrame:
    """Descarga datos históricos usando yfinance"""
    try:
        import yfinance as yf
    except ImportError:
        print("Error: yfinance no instalado. Instálalo con: pip install yfinance")
        sys.exit(1)
        
    end = datetime.now()
    start = end - timedelta(days=years*365)
    
    print(f"Descargando {years} años de datos para {symbol}...")
    df = yf.download(symbol, start=start, end=end, progress=False)
    
    if df.empty:
        print(f"No se encontraron datos para {symbol}")
        sys.exit(1)
        
    # Renombrar columnas a nuestro estándar
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
    df.columns = ['open', 'high', 'low', 'close', 'volume']
    df.index.name = 'timestamp'
    
    print(f"Datos descargados: {len(df)} barras")
    print(f"Rango: {df.index[0].date()} → {df.index[-1].date()}")
    return df


def load_csv(filepath: str) -> pd.DataFrame:
    """Carga datos desde CSV local"""
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
    parser = argparse.ArgumentParser(description='Backtester EMA 3/22 + Zonas Mitigadas')
    parser.add_argument('csv', nargs='?', help='Archivo CSV con datos (opcional si usas --download)')
    parser.add_argument('--download', metavar='SYMBOL', help='Descargar datos de yfinance (ej: BTC-USD)')
    parser.add_argument('--years', type=int, default=5, help='Años de datos a descargar (default: 5)')
    parser.add_argument('--fast', type=int, default=3, help='EMA rápida (default: 3)')
    parser.add_argument('--slow', type=int, default=22, help='EMA lenta (default: 22)')
    parser.add_argument('--window', type=int, default=10, help='Ventana post-mitigación (default: 10)')
    
    args = parser.parse_args()
    
    # Cargar o descargar datos
    if args.download:
        df = download_data(args.download, years=args.years)
        csv_name = f"{args.download.replace('-','_')}_{args.years}y.csv"
        df.to_csv(csv_name)
        print(f"Datos guardados en {csv_name}")
    elif args.csv:
        df = load_csv(args.csv)
    else:
        print("Error: Debes especificar un archivo CSV o usar --download SYMBOL")
        parser.print_help()
        sys.exit(1)
        
    # Configurar estrategia
    params = {
        'fast_ema': args.fast,
        'slow_ema': args.slow,
        'bars_window': args.window,
        'swing_len': 16,
        'liq_tolerance': 0.5,
        'sens_fvg': 0.5,
        'atr_len': 14,
        'tp_mult': 3.0,
        'sl_mult': 2.0,
        'use_liq': True,
        'use_fvg': True
    }
    
    print("\n" + "="*70)
    print("CONFIGURACIÓN ESTRATEGIA")
    print("="*70)
    for k, v in params.items():
        print(f"{k}: {v}")
    print("-"*70)
    
    # Ejecutar backtest
    strat = EstrategiaEMAZonas(**params)
    results = strat.run_backtest(df)
    
    # Mostrar resultados
    strat.print_stats(results)
    
    # Opción de guardar trades detallados
    save = input("\n¿Guardar trades en CSV? (s/N): ").strip().lower()
    if save == 's':
        trades_df = pd.DataFrame(results['trades'])
        trades_df.to_csv('trades_ema_zonas.csv', index=False)
        print("Trades guardados en trades_ema_zonas.csv")
        
        signals_df = results['df'][['close','ema_fast','ema_slow','signal','zone_mitigated']].copy()
        signals_df.to_csv('signals_ema_zonas.csv')
        print("Señales guardadas en signals_ema_zonas.csv")


if __name__ == "__main__":
    main()
