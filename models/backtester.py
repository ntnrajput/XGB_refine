\
# models/backtester.py - Improved backtesting engine
# Updated: robust prediction input sanitation (duplicate columns, NaN/Inf coercion)
#          and targeted warning suppression for cleaner logs.
#          Function names and functionality preserved or expanded; no line reductions.

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings("ignore", message="invalid value encountered in divide")
warnings.filterwarnings("ignore", message="Data must be 1-dimensional")

from models.trainer import ModelLoader
from utils.logger import get_logger
from utils.exceptions import ModelError

logger = get_logger(__name__)


class Backtester:
    """Backtest trading strategies"""

    def __init__(self):
        self.trades = []

    def run(self, df, model_path, initial_capital, swing_config, save_charts=True):
        """
        Run backtest simulation.

        Args:
            df: Data with features (should have columns: date, symbol, close, open, high, low, volume)
            model_path: Path to trained model
            initial_capital: Starting capital
            swing_config: Trading config object with attributes:
                - max_hold_days: Maximum holding period
                - target_pct: Target profit percentage
                - stop_loss_pct: Stop loss percentage
            save_charts: Save result charts

        Returns:
            Results dictionary with metrics and trades
        """
        logger.info("[backtester.py] 📈 Starting backtest...")

        # Validate input data
        required_cols = ['date', 'symbol', 'close']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ModelError(f"Missing required columns: {missing_cols}")

        # Load model
        model_bundle = ModelLoader.load(model_path)

        # Get config values with defaults
        max_hold_days = getattr(swing_config, 'max_hold_days', 10)
        target_pct = getattr(swing_config, 'target_pct', 0.05)  # 5%
        stop_loss_pct = getattr(swing_config, 'stop_loss_pct', 0.03)  # 3%

        print('-'*50)
        print(max_hold_days,target_pct,stop_loss_pct)
        print('-'*50)

        logger.info(f"[backtester.py] Backtest parameters:")
        logger.info(f"   Initial capital: ₹{initial_capital:,.0f}")
        logger.info(f"   Max hold days: {max_hold_days}")
        logger.info(f"   Target: {target_pct*100:.1f}%")
        logger.info(f"   Stop loss: {stop_loss_pct*100:.1f}%")
        logger.info(f"   Date range: {df['date'].min()} to {df['date'].max()}")
        logger.info(f"   Total rows: {len(df):,}")
        logger.info(f"   Symbols: {df['symbol'].nunique()}")

        # Simulate trading
        trades = self._simulate_trading(
            df, 
            model_bundle, 
            initial_capital,
            max_hold_days,
            target_pct,
            stop_loss_pct
        )

        # Calculate metrics
        metrics = self._calculate_metrics(trades, initial_capital, df)

        results = {
            'metrics': metrics,
            'trades': trades
        }

        logger.info(f"[backtester.py] ✅ Backtest complete")

        try:
            trades_df = pd.DataFrame(trades)
            out_path = Path(model_path).parent / f"backtest_trades_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            trades_df.to_csv(out_path, index=False)
            logger.info(f"[backtester.py] 💾 Saved trades to: {out_path}")
        except Exception as e:
            logger.warning(f"[backtester.py] ⚠️ Failed to save trades file: {e}")

        return results


    def _simulate_trading(
        self, 
        df: pd.DataFrame,
        model_bundle: Dict,
        initial_capital: float,
        max_hold_days: int,
        target_pct: float,
        stop_loss_pct: float
    ) -> List[Dict]:
        """Simulate trading based on model predictions"""

        logger.info("[backtester.py] 🔮 Getting model predictions...")

        # Get predictions for all data
        try:
            # Ensure df is not carrying duplicate-named columns into prediction
            # (extra safety; ModelLoader.predict also handles this)
            if isinstance(df, pd.DataFrame) and df.columns.duplicated(keep="first").any():
                dup_cols = df.columns[df.columns.duplicated(keep="first")].tolist()
                logger.warning(f"[backtester.py] ⚠️  Dropping duplicate columns prior to predict: {dup_cols[:10]}{' ...' if len(dup_cols)>10 else ''}")
                df = df.loc[:, ~df.columns.duplicated(keep="first")].copy()

            predictions, probabilities = ModelLoader.predict(model_bundle, df)
            df = df.copy()
            # probabilities may be shape (n,2) for binary classifiers -> take [:,1]
            prob_1 = probabilities[:, 1] if probabilities.ndim == 2 and probabilities.shape[1] > 1 else probabilities.ravel()
            df['prediction'] = predictions
            df['probability'] = prob_1  # Probability of class 1 (buy signal)
        except Exception as e:
            logger.error(f"[backtester.py] ❌ Prediction failed: {e}")
            raise

        logger.info(f"[backtester.py] ✅ Predictions generated")
        total_signals = int(np.sum(df['prediction'].to_numpy()))
        logger.info(f"   Total signals: {total_signals:,} ({(total_signals/len(df))*100:.1f}%)")

        # Group by symbol for easier processing
        logger.info("[backtester.py] 🔄 Simulating trades...")

        trades = []
        capital = initial_capital
        position_size_pct = 0.1  # Use 10% of capital per trade
        min_probability = 0.77  # Minimum probability to enter trade

        # Sort by date to simulate chronological trading
        df = df.sort_values(['symbol', 'date']).reset_index(drop=True)

        # Process each symbol separately
        symbols = df['symbol'].unique()

        for symbol_idx, symbol in enumerate(symbols):
            symbol_df = df[df['symbol'] == symbol].reset_index(drop=True)

            # Track position for this symbol
            in_position = False
            entry_idx = None

            i = 0  # manual index to allow skipping forward
            while i < len(symbol_df):
                # Skip if we don't have enough future data
                if i >= len(symbol_df) - max_hold_days:
                    break

                # Check for entry signal
                if not in_position:
                    if (int(symbol_df.at[i, 'prediction']) == 1 and 
                        float(symbol_df.at[i, 'probability']) >= float(min_probability)):

                        # Enter trade
                        entry_idx = i
                        entry_date = symbol_df.at[i, 'date']
                        entry_price = float(symbol_df.at[i, 'close'])

                        # Calculate position size
                        position_value = capital * position_size_pct
                        shares = position_value / max(entry_price, 1e-9)

                        in_position = True

                        # Look for exit
                        exit_idx = None
                        exit_reason = 'max_hold'

                        for j in range(i + 1, min(i + max_hold_days + 1, len(symbol_df))):
                            current_price = float(symbol_df.at[j, 'close'])
                            current_return = (current_price - entry_price) / max(entry_price, 1e-9)

                            # Check exit conditions
                            if current_return >= target_pct:
                                exit_idx = j
                                exit_reason = 'target'
                                break
                            elif current_return <= -stop_loss_pct:
                                exit_idx = j
                                exit_reason = 'stop_loss'
                                break

                        # If no exit triggered, exit at max hold
                        if exit_idx is None:
                            exit_idx = min(i + max_hold_days, len(symbol_df) - 1)
                            exit_reason = 'max_hold'

                        # Execute exit
                        exit_date = symbol_df.at[exit_idx, 'date']
                        exit_price = float(symbol_df.at[exit_idx, 'close'])

                        pnl = (exit_price - entry_price) * shares
                        pnl_pct = (exit_price - entry_price) / max(entry_price, 1e-9) * 100.0

                        capital += pnl

                        # Record trade
                        trades.append({
                            'symbol': symbol,
                            'entry_date': entry_date,
                            'exit_date': exit_date,
                            'entry_price': float(entry_price),
                            'exit_price': float(exit_price),
                            'shares': float(shares),
                            'pnl': float(pnl),
                            'pnl_pct': float(pnl_pct),
                            'exit_reason': exit_reason,
                            'capital_after': float(capital),
                            'hold_days': int(exit_idx - i)
                        })

                        in_position = False

                        # Skip ahead to avoid overlapping trades
                        i = exit_idx + 1
                        continue

                i += 1  # advance if no trade taken

            # Progress update every 100 symbols
            if (symbol_idx + 1) % 100 == 0:
                logger.info(f"   Processed {symbol_idx + 1}/{len(symbols)} symbols, {len(trades)} trades so far")

        logger.info(f"[backtester.py] ✅ Simulation complete: {len(trades)} trades executed")

        return trades

    def _calculate_metrics(
        self, 
        trades: List[Dict], 
        initial_capital: float,
        df: pd.DataFrame
    ) -> Dict:
        """Calculate performance metrics"""

        if not trades:
            logger.warning("[backtester.py] ⚠️  No trades executed")
            return {
                'initial_capital': initial_capital,
                'final_capital': initial_capital,
                'total_return_pct': 0,
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0,
                'avg_win': 0,
                'avg_loss': 0,
                'profit_factor': 0,
                'max_drawdown': 0,
                'avg_hold_days': 0,
                'cagr': 0,
                'sharpe_ratio': 0
            }

        final_capital = trades[-1]['capital_after']

        winning_trades = [t for t in trades if t['pnl'] > 0]
        losing_trades = [t for t in trades if t['pnl'] <= 0]

        total_win = sum(t['pnl'] for t in winning_trades) if winning_trades else 0
        total_loss = abs(sum(t['pnl'] for t in losing_trades)) if losing_trades else 0

        # Calculate max drawdown
        capital_curve = [initial_capital] + [t['capital_after'] for t in trades]
        peak = capital_curve[0]
        max_dd = 0.0
        for cap in capital_curve:
            if cap > peak:
                peak = cap
            dd = (peak - cap) / peak * 100.0 if peak > 0 else 0.0
            if dd > max_dd:
                max_dd = dd

        # Calculate average holding days
        avg_hold_days = float(np.mean([t['hold_days'] for t in trades])) if trades else 0.0

        # Calculate CAGR (simplified)
        start_date = pd.to_datetime(df['date'].min())
        end_date = pd.to_datetime(df['date'].max())
        years = (end_date - start_date).days / 365.25
        if years > 0:
            cagr = (pow(final_capital / initial_capital, 1 / years) - 1) * 100.0
        else:
            cagr = 0.0

        # Calculate Sharpe ratio (simplified)
        returns = np.array([t['pnl_pct'] for t in trades], dtype=float)
        if returns.size > 1 and float(np.std(returns)) > 0:
            sharpe = float(np.mean(returns) / np.std(returns) * np.sqrt(252))
        else:
            sharpe = 0.0

        metrics = {
            'initial_capital': float(initial_capital),
            'final_capital': float(final_capital),
            'total_return': float(final_capital - initial_capital),
            'total_return_pct': float((final_capital - initial_capital) / initial_capital * 100.0),
            'total_trades': int(len(trades)),
            'winning_trades': int(len(winning_trades)),
            'losing_trades': int(len(losing_trades)),
            'win_rate': float(len(winning_trades) / len(trades) * 100.0) if trades else 0.0,
            'avg_win': float(total_win / len(winning_trades)) if winning_trades else 0.0,
            'avg_loss': float(total_loss / len(losing_trades)) if losing_trades else 0.0,
            'profit_factor': float(total_win / total_loss) if total_loss > 0 else 0.0,
            'max_drawdown': float(max_dd),
            'avg_hold_days': float(avg_hold_days),
            'cagr': float(cagr),
            'sharpe_ratio': float(sharpe)
        }

        return metrics


def run_backtest(pipeline, model_config, capital, start_date):
    """Convenience function"""
    backtester = Backtester()
    return backtester


__all__ = ['Backtester', 'run_backtest']
