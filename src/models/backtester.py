"""Backtesting Engine."""

import logging
from dataclasses import dataclass, field
from typing import List

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(name)-25s | %(levelname)-8s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


@dataclass
class Trade:
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    entry_price: float
    exit_price: float
    direction: str
    quantity: float
    pnl: float
    pnl_pct: float


@dataclass
class BacktestResults:
    initial_capital: float
    final_capital: float
    total_return: float
    total_return_pct: float
    n_trades: int
    n_wins: int
    n_losses: int
    win_rate: float
    avg_win: float
    avg_loss: float
    largest_win: float
    largest_loss: float
    sharpe_ratio: float
    max_drawdown: float
    max_drawdown_pct: float
    profit_factor: float
    equity_curve: pd.DataFrame
    trades: List[Trade] = field(default_factory=list)


class Backtester:
    def __init__(
        self,
        initial_capital: float = 10_000_000,
        position_size_pct: float = 0.10,
        taker_fee: float = 0.003,
        confidence_threshold: float = 0.50,
    ) -> None:
        self.initial_capital = initial_capital
        self.position_size_pct = position_size_pct
        self.taker_fee = taker_fee
        self.confidence_threshold = confidence_threshold
    
    def run(self, prices: pd.DataFrame, signals: pd.DataFrame) -> BacktestResults:
        logger.info(f"Running backtest on {len(prices)} candles")
        logger.info(f"Initial capital: Rp {self.initial_capital:,.0f}")
        
        df = self._prepare_data(prices, signals)
        
        cash = self.initial_capital
        position = 0.0
        entry_price = 0.0
        entry_time = None
        
        trades = []
        equity_history = []
        
        for idx, row in df.iterrows():
            current_price = row['close']
            current_time = row['timestamp']
            signal = row['signal']
            confidence = row['confidence']
            
            equity = cash + (position * current_price)
            equity_history.append({
                'timestamp': current_time,
                'equity': equity,
                'cash': cash,
                'position_value': position * current_price,
                'position_qty': position,
            })
            
            if confidence < self.confidence_threshold:
                continue
            
            if signal == 'BUY' and position == 0:
                trade_amount = cash * self.position_size_pct
                fee = trade_amount * self.taker_fee
                quantity = (trade_amount - fee) / current_price
                cash -= trade_amount
                position = quantity
                entry_price = current_price
                entry_time = current_time
            
            elif signal == 'SELL' and position > 0:
                trade_value = position * current_price
                fee = trade_value * self.taker_fee
                proceeds = trade_value - fee
                cost_basis = position * entry_price
                pnl = proceeds - cost_basis
                pnl_pct = (pnl / cost_basis) * 100
                
                trades.append(Trade(
                    entry_time=entry_time,
                    exit_time=current_time,
                    entry_price=entry_price,
                    exit_price=current_price,
                    direction='LONG',
                    quantity=position,
                    pnl=pnl,
                    pnl_pct=pnl_pct,
                ))
                
                cash += proceeds
                position = 0.0
                entry_price = 0.0
                entry_time = None
        
        if position > 0:
            final_row = df.iloc[-1]
            final_price = final_row['close']
            trade_value = position * final_price
            fee = trade_value * self.taker_fee
            proceeds = trade_value - fee
            cost_basis = position * entry_price
            pnl = proceeds - cost_basis
            pnl_pct = (pnl / cost_basis) * 100
            
            trades.append(Trade(
                entry_time=entry_time,
                exit_time=final_row['timestamp'],
                entry_price=entry_price,
                exit_price=final_price,
                direction='LONG',
                quantity=position,
                pnl=pnl,
                pnl_pct=pnl_pct,
            ))
            
            cash += proceeds
            position = 0.0
        
        equity_df = pd.DataFrame(equity_history)
        results = self._calculate_metrics(cash, equity_df, trades)
        
        logger.info(f"Backtest complete: {results.n_trades} trades executed")
        return results
    
    def _prepare_data(self, prices, signals):
        df = prices[['timestamp', 'close']].copy().reset_index(drop=True)
        if len(signals) == len(df):
            df['signal'] = signals['signal'].values
            df['confidence'] = signals['confidence'].values
        else:
            raise ValueError(f"Length mismatch: prices={len(df)}, signals={len(signals)}")
        return df
    
    def _calculate_metrics(self, final_cash, equity_df, trades):
        total_return = final_cash - self.initial_capital
        total_return_pct = (total_return / self.initial_capital) * 100
        
        n_trades = len(trades)
        wins = [t for t in trades if t.pnl > 0]
        losses = [t for t in trades if t.pnl <= 0]
        n_wins = len(wins)
        n_losses = len(losses)
        win_rate = (n_wins / n_trades * 100) if n_trades > 0 else 0.0
        
        avg_win = np.mean([t.pnl for t in wins]) if wins else 0.0
        avg_loss = np.mean([t.pnl for t in losses]) if losses else 0.0
        largest_win = max((t.pnl for t in wins), default=0.0)
        largest_loss = min((t.pnl for t in losses), default=0.0)
        
        equity_df['returns'] = equity_df['equity'].pct_change()
        returns = equity_df['returns'].dropna()
        if len(returns) > 1 and returns.std() > 0:
            sharpe = (returns.mean() / returns.std()) * np.sqrt(24 * 365)
        else:
            sharpe = 0.0
        
        equity_df['running_max'] = equity_df['equity'].cummax()
        equity_df['drawdown'] = equity_df['equity'] - equity_df['running_max']
        equity_df['drawdown_pct'] = (
            equity_df['drawdown'] / equity_df['running_max'] * 100
        )
        max_drawdown = equity_df['drawdown'].min()
        max_drawdown_pct = equity_df['drawdown_pct'].min()
        
        total_profit = sum(t.pnl for t in wins) if wins else 0.0
        total_loss = abs(sum(t.pnl for t in losses)) if losses else 1.0
        profit_factor = total_profit / total_loss if total_loss > 0 else 0.0
        
        return BacktestResults(
            initial_capital=self.initial_capital,
            final_capital=final_cash,
            total_return=total_return,
            total_return_pct=total_return_pct,
            n_trades=n_trades,
            n_wins=n_wins,
            n_losses=n_losses,
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            largest_win=largest_win,
            largest_loss=largest_loss,
            sharpe_ratio=sharpe,
            max_drawdown=max_drawdown,
            max_drawdown_pct=max_drawdown_pct,
            profit_factor=profit_factor,
            equity_curve=equity_df,
            trades=trades,
        )
    
    def calculate_buy_and_hold(self, prices: pd.DataFrame) -> dict:
        first_price = prices['close'].iloc[0]
        last_price = prices['close'].iloc[-1]
        
        fee = self.initial_capital * self.taker_fee
        quantity = (self.initial_capital - fee) / first_price
        
        final_value = quantity * last_price
        sell_fee = final_value * self.taker_fee
        final_capital = final_value - sell_fee
        
        return {
            'initial_capital': self.initial_capital,
            'final_capital': final_capital,
            'total_return': final_capital - self.initial_capital,
            'total_return_pct': (
                (final_capital - self.initial_capital)
                / self.initial_capital * 100
            ),
        }