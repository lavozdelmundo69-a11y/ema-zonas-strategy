#!/usr/bin/env python3
"""
Backtester para Estrategia: EMA 3/22 + Zonas Mitigadas

Señal de entrada:
- Cruce de EMA 3 sobre EMA 22 (long) o bajo (short)
- Previamente (ventana configurable) se mitigó una zona de:
  * Liquidez (swing high/low)
  * Zona de Interés / FVG

Salida:
- Take Profit: X veces ATR
- Stop Loss: Y veces ATR

Uso:
    python backtester.py datos.csv
    (CSV con columnas: timestamp,open,high,low,close,volume)
"""

import sys
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Optional

class EstrategiaEMAZonas:
    def __init__(self, fast_ema=3, slow_ema=22, bars_window=10,
                 swing_len=16, liq_tolerance=0.5,
                 sens_fvg=0.5, atr_len=14, tp_mult=3.0, sl_mult=2.0):
        # Parámetros
        self.fast_ema = fast_ema
        self.slow_ema = slow_ema
        self.bars_window = bars_window
        self.swing_len = swing_len
        self.liq_tolerance = liq_tolerance / 100.0
        self.sens_fvg = sens_fvg / 100.0
        self.atr_len = atr_len
        self.tp_mult = tp_mult
        self.sl_mult = sl_mult
        
        # Arrays para tracking zonas (simulación de arrays de Pine)
        self.liq_highs = []        # [(price, bar_idx, mitigated, mitig_bar), ...]
        self.liq_lows = []
        self.fvg_zones = []        # [(top, bot, origin_bar, mitigated, mitig_bar, is_bull), ...]
        
    def detect_swing_highs_lows(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detecta swing highs y lows usando pivots"""
        swing_high = df['high'].rolling(self.swing_len, center=True).apply(
            lambda x: x[self.swing_len//2] == max(x), raw=True
        )
        swing_low = df['low'].rolling(self.swing_len, center=True).apply(
            lambda x: x[self.swing_len//2] == min(x), raw=True
        )
        df['swing_high'] = swing_high.fillna(0).astype(bool)
        df['swing_low'] = swing_low.fillna(0).astype(bool)
        return df
    
    def add_liquidity_zone(self, price: float, bar_idx: int, is_high: bool):
        """Agrega una nueva zona de liquidez"""
        arr = self.liq_highs if is_high else self.liq_lows
        arr.append({
            'price': price,
            'bar_idx': bar_idx,
            'mitigated': False,
            'mitig_bar': -1
        })
        
    def check_liquidity_mitigation(self, close: float, bar_idx: int):
        """Verifica si alguna zona de liquidez se mitiga con el cierre actual"""
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
        """Verifica mitigación de FVGs"""
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
                
    def add_fvg_zone(self, df: pd.DataFrame, i: int, is_bull: bool):
        """Agrega un FVG válido"""
        if is_bull:
            top, bot = df.iloc[i]['low'], df.iloc[i-2]['high']
        else:
            top, bot = df.iloc[i-2]['low'], df.iloc[i]['high']
            
        size_pct = abs(top - bot) / df.iloc[i]['close'] * 100
        if size_pct >= self.sens_fvg * 100:
            self.fvg_zones.append({
                'top': top,
                'bot': bot,
                'origin_bar': i-1,  # bar where FVG formed (1 bar ago)
                'mitigated': False,
                'mitig_bar': -1,
                'is_bull': is_bull
            })
            
    def recent_zone_mitigated(self, bar_idx: int) -> bool:
        """Verifica si alguna zona se mitigó en la ventana de barras"""
        window_start = bar_idx - self.bars_window
        
        # Chequear zonas de liquidez
        for arr in [self.liq_highs, self.liq_lows]:
            for zone in arr:
                if zone['mitigated'] and zone['mitig_bar'] >= window_start:
                    return True
                    
        # Chequear FVGs
        for zone in self.fvg_zones:
            if zone['mitigated'] and zone['mitig_bar'] >= window_start:
                return True
                
        return False
        
    def calculate_indicators(self, df: pd.DataFrame):
        """Calcula EMAs, ATR y detecta zonas"""
        # EMAs
        df['ema_fast'] = df['close'].ewm(span=self.fast_ema, adjust=False).mean()
        df['ema_slow'] = df['close'].ewm(span=self.slow_ema, adjust=False).mean()
        
        # ATR
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        df['atr'] = true_range.rolling(window=self.atr_len).mean()
        
        # Detectar pivots
        df = self.detect_swing_highs_lows(df)
        
        # Detectar FVG simple en cada barra
        for i in range(2, len(df)):
            # FVG Alcista
            if df.iloc[i]['low'] > df.iloc[i-2]['high'] and df.iloc[i-1]['close'] > df.iloc[i-1]['open']:
                self.add_fvg_zone(df, i, is_bull=True)
            # FVG Bajista
            if df.iloc[i]['high'] < df.iloc[i-2]['low'] and df.iloc[i-1]['close'] < df.iloc[i-1]['open']:
                self.add_fvg_zone(df, i, is_bull=False)
                
        return df
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Genera señales de entrada"""
        df['signal'] = 0  # 0=nada, 1=long, -1=short
        
        for i in range(self.slow_ema, len(df)):
            # Actualizar zonas con el cierre de esta barra
            close_i = df.iloc[i]['close']
            self.check_liquidity_mitigation(close_i, i)
            self.check_fvg_mitigation(close_i, i)
            
            # Verificar cruces
            prev_fast = df.iloc[i-1]['ema_fast']
            prev_slow = df.iloc[i-1]['ema_slow']
            curr_fast = df.iloc[i]['ema_fast']
            curr_slow = df.iloc[i]['ema_slow']
            
            golden = (prev_fast <= prev_slow) and (curr_fast > curr_slow)
            dead = (prev_fast >= prev_slow) and (curr_fast < curr_slow)
            
            # ¿Hay mitigación reciente?
            zone_ok = self.recent_zone_mitigated(i)
            
            if golden and zone_ok:
                df.at[df.index[i], 'signal'] = 1
            elif dead and zone_ok:
                df.at[df.index[i], 'signal'] = -1
                
        return df
    
    def run_backtest(self, df: pd.DataFrame, initial_capital=10000) -> Dict:
        """Ejecuta el backtest completo"""
        df = self.calculate_indicators(df.copy())
        df = self.generate_signals(df)
        
        capital = initial_capital
        position = 0  # 0=sin posición, 1=long, -1=short
        entry_price = 0.0
        trades = []
        
        for i in range(len(df)):
            signal = df.iloc[i]['signal']
            close = df.iloc[i]['close']
            atr = df.iloc[i]['atr']
            
            # Si no tenemos posición y hay señal, entramos
            if position == 0 and signal != 0:
                position = signal
                entry_price = close
                tp_price = close + signal * self.tp_mult * atr
                sl_price = close - signal * self.sl_mult * atr
                entry_bar = i
                
            # Si tenemos posición, verificar TP/SL
            elif position != 0:
                # Chequear Stop Loss
                if position == 1 and close <= sl_price:
                    # Stop loss en long
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
                        'reason': 'sl'
                    })
                    position = 0
                    
                elif position == -1 and close >= sl_price:
                    # Stop loss en short
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
                        'reason': 'sl'
                    })
                    position = 0
                    
                # Chequear Take Profit
                elif position == 1 and close >= tp_price:
                    # TP en long
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
                        'reason': 'tp'
                    })
                    position = 0
                    
                elif position == -1 and close <= tp_price:
                    # TP en short
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
                        'reason': 'tp'
                    })
                    position = 0
                    
        # Cerrar posición abierta al final
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
                'reason': 'eof'
            })
            
        return {
            'final_capital': capital,
            'total_return_pct': (capital / initial_capital - 1) * 100,
            'trades': trades,
            'df': df  # Para análisis posterior
        }
    
    def print_stats(self, results: Dict):
        """Imprime estadísticas del backtest"""
        trades = results['trades']
        if not trades:
            print("No se generaron trades.")
            return
            
        total_trades = len(trades)
        winning_trades = sum(1 for t in trades if t['pnl'] > 0)
        losing_trades = sum(1 for t in trades if t['pnl'] < 0)
        win_rate = winning_trades / total_trades * 100
        
        total_pnl = sum(t['pnl'] for t in trades)
        avg_win = np.mean([t['pnl'] for t in trades if t['pnl'] > 0]) if winning_trades else 0
        avg_loss = np.mean([t['pnl'] for t in trades if t['pnl'] < 0]) if losing_trades else 0
        profit_factor = abs(sum(t['pnl'] for t in trades if t['pnl'] > 0) / 
                           sum(t['pnl'] for t in trades if t['pnl'] < 0)) if losing_trades else np.inf
        
        print("="*60)
        print("ESTADÍSTICAS BACKTEST")
        print("="*60)
        print(f"Capital inicial:     {results.get('initial_capital', 10000):,.2f}")
        print(f"Capital final:       {results['final_capital']:,.2f}")
        print(f"Retorno total:       {results['total_return_pct']:.2f}%")
        print(f"Total trades:        {total_trades}")
        print(f"Win rate:            {win_rate:.2f}%")
        print(f"Ganadores:          {winning_trades}")
        print(f"Perdedores:         {losing_trades}")
        print(f"Ganancia promedio:  {avg_win:.2f}")
        print(f"Pérdida promedio:   {avg_loss:.2f}")
        print(f"Profit Factor:      {profit_factor:.2f}")
        print("="*60)
        
        # Conteo por razón de salida
        reasons = {}
        for t in trades:
            reason = t['reason']
            reasons[reason] = reasons.get(reason, 0) + 1
        print("Razones de salida:")
        for reason, count in reasons.items():
            print(f"  {reason}: {count} ({count/total_trades*100:.1f}%)")
            
        # Tipos de señal
        longs = sum(1 for t in trades if t['type']=='long')
        shorts = sum(1 for t in trades if t['type']=='short')
        print(f"Longs: {longs}, Shorts: {shorts}")
        print("="*60)


def load_data(filepath: str) -> pd.DataFrame:
    """Carga datos desde CSV (o Yahoo Finance si no hay CSV)"""
    try:
        df = pd.read_csv(filepath, parse_dates=['timestamp'])
        df.set_index('timestamp', inplace=True)
        print(f"Datos cargados desde {filepath}: {len(df)} filas")
    except FileNotFoundError:
        print(f"Archivo {filepath} no encontrado. Necesitas un CSV con columnas:")
        print("timestamp,open,high,low,close,volume")
        print("O modifica el script para usar yfinance.")
        sys.exit(1)
    except Exception as e:
        print(f"Error cargando CSV: {e}")
        print("Asegúrate de tener columnas: timestamp,open,high,low,close,volume")
        sys.exit(1)
        
    return df


def main():
    """Función principal"""
    if len(sys.argv) < 2:
        print("Uso: python backtester.py <archivo.csv> [parametros]")
        print("Ejemplo: python backtester.py datos.csv --fast 3 --slow 22")
        sys.exit(1)
        
    filepath = sys.argv[1]
    df = load_data(filepath)
    
    # Parámetros por defecto (puedes agregar argparse si quieres personalizar)
    params = {
        'fast_ema': 3,
        'slow_ema': 22,
        'bars_window': 10,
        'swing_len': 16,
        'liq_tolerance': 0.5,
        'sens_fvg': 0.5,
        'atr_len': 14,
        'tp_mult': 3.0,
        'sl_mult': 2.0
    }
    
    print("\nEstrategia: EMA 3/22 + Zonas Mitigadas")
    print(f"Parámetros: {params}")
    print("-"*60)
    
    # Ejecutar backtest
    strat = EstrategiaEMAZonas(**params)
    results = strat.run_backtest(df, initial_capital=10000)
    results['initial_capital'] = 10000
    
    # Mostrar resultados
    strat.print_stats(results)
    
    # Preguntar si guardar resultados detallados
    save = input("\n¿Guardar resultados en CSV? (s/N): ").strip().lower()
    if save == 's':
        trades_df = pd.DataFrame(results['trades'])
        trades_df.to_csv('trades_result.csv', index=False)
        print("Trades guardados en trades_result.csv")
        
        signals_df = results['df'][['close','ema_fast','ema_slow','signal']].copy()
        signals_df.to_csv('signals_result.csv')
        print("Señales guardadas en signals_result.csv")


if __name__ == "__main__":
    main()
