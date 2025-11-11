"""
LLM Strategy Streaming Backtest API - Server-Sent Events (SSE)
Provides real-time progress updates and result streaming
"""

import asyncio
import json
import os
import queue
import threading
import time
from datetime import datetime, timedelta
from typing import Generator, Optional

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse

from ....backtesting.engine import BacktestConfig, BacktestEngine
from ....llm.strategies.base import StrategyConfig
from ....llm.strategies.llm_strategy import LLMSmartStrategy  # 切換回原版
from ....utils.stock_data import StockService

router = APIRouter()


def safe_json_serialize(obj):
    """
    安全的 JSON 序列化函數，處理各種不可序列化的對象
    """
    if isinstance(obj, (pd.DataFrame, pd.Series)):
        # DataFrame 轉換為字典列表
        return obj.to_dict("records") if hasattr(obj, "to_dict") else str(obj)
    elif isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    elif isinstance(obj, datetime):
        return obj.isoformat()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif hasattr(obj, "to_dict"):
        # 如果對象有 to_dict 方法，使用它
        return obj.to_dict()
    elif hasattr(obj, "__dict__"):
        # 如果是自定義對象，嘗試序列化其屬性
        return {k: safe_json_serialize(v) for k, v in obj.__dict__.items()}
    else:
        return str(obj)


def safe_json_dumps(obj):
    """
    安全的 JSON dumps，預處理不可序列化的對象
    """

    def json_serializer(o):
        return safe_json_serialize(o)

    return json.dumps(obj, default=json_serializer, ensure_ascii=False)


@router.get("/llm-backtest-stream")
async def stream_llm_backtest(
    symbol: str = Query(..., description="Stock symbol"),
    period: str = Query("1y", description="Backtest period"),
    max_position_size: float = Query(0.3, description="Maximum position size"),
    stop_loss: float = Query(0.05, description="Stop loss percentage"),
    take_profit: float = Query(0.1, description="Take profit percentage"),
):
    """
    Streaming LLM strategy backtest - using Server-Sent Events
    Unlimited capital mode: Uses fixed large capital, focuses on pure trading P&L calculation
    """
    # Use unlimited capital mode (100M USD)
    initial_capital = 100000000.0
    try:

        def generate_backtest_stream() -> Generator[str, None, None]:
            """生成回測進度流"""

            # 創建消息隊列用於線程間通信
            message_queue = queue.Queue()

            def progress_callback(
                day: int,
                total_days: int,
                event_type: str,
                message: str,
                extra_data: dict = None,
            ):
                """進度回調函數"""
                progress_data = {
                    "type": "trading_progress",
                    "day": day,
                    "total_days": total_days,
                    "progress": round((day / total_days) * 100, 1),
                    "event_type": event_type,
                    "message": message,
                }

                # 如果有額外數據（如P&L信息），添加到進度數據中
                if extra_data:
                    progress_data.update(extra_data)

                message_queue.put(progress_data)

            def run_backtest():
                """Run backtest in separate thread"""
                try:
                    # 1. Fetch stock data
                    message_queue.put(
                        {
                            "type": "progress",
                            "step": "data_loading",
                            "message": f"Fetching {symbol} stock data...",
                        }
                    )

                    stock_service = StockService()
                    stock_data_list = stock_service.get_market_data(symbol, period)

                    if not stock_data_list or len(stock_data_list) < 30:
                        message_queue.put(
                            {"type": "error", "message": "Insufficient stock data, cannot run backtest"}
                        )
                        return

                    # Convert to DataFrame, backtest engine needs DataFrame format
                    stock_data = pd.DataFrame(stock_data_list)
                    # Set date as index and ensure column names match expectations
                    stock_data["date"] = pd.to_datetime(stock_data["date"])
                    stock_data.set_index("date", inplace=True)
                    # Rename columns to standard format (lowercase, as strategy and engine expect lowercase column names)
                    stock_data.columns = ["open", "high", "low", "close", "volume"]

                    message_queue.put(
                        {
                            "type": "progress",
                            "step": "data_loaded",
                            "message": f"Successfully fetched {len(stock_data)} days of data",
                        }
                    )

                    # 2. Initialize strategy
                    message_queue.put(
                        {
                            "type": "progress",
                            "step": "strategy_init",
                            "message": "Initializing LLM strategy...",
                        }
                    )

                    # 創建 LLM 策略配置 - 使用 .env 中的固定模型
                    strategy_config = StrategyConfig(
                        name="LLM Trading Strategy",
                        description="AI-powered trading strategy with real-time analysis",
                        parameters={
                            "initial_capital": initial_capital,
                            "max_position_size": max_position_size,
                            "stop_loss": stop_loss,
                            "take_profit": take_profit,
                            "progress_callback": progress_callback,
                            # Enable backtest logging
                            "enable_logging": True,
                            "log_path": os.path.join(
                                "backend", "data", "backtest_logs.db"
                            ),
                            "session_id": f"api_session_{symbol}_{int(time.time())}",
                        },
                    )

                    strategy = LLMSmartStrategy(strategy_config)

                    # 3. Start backtest
                    message_queue.put(
                        {
                            "type": "progress",
                            "step": "backtest_start",
                            "message": "Starting backtest execution...",
                        }
                    )

                    # 建立回測設定
                    backtest_config = BacktestConfig(initial_capital=initial_capital)
                    backtest_engine = BacktestEngine(backtest_config)

                    # 執行回測
                    backtest_result = backtest_engine.run_backtest(
                        stock_data=stock_data,
                        strategy=strategy,
                        initial_cash=initial_capital,
                        symbol=symbol,
                    )

                    # 4. Send results
                    message_queue.put(
                        {
                            "type": "progress",
                            "step": "analysis",
                            "message": "Analyzing backtest results...",
                        }
                    )

                    # 將 DataFrame 轉換為可 JSON 序列化的格式
                    stock_data_json = []
                    for date, row in stock_data.iterrows():
                        stock_data_json.append(
                            {
                                "timestamp": date.strftime("%Y-%m-%d"),
                                "open": float(row["open"]),
                                "high": float(row["high"]),
                                "low": float(row["low"]),
                                "close": float(row["close"]),
                                "volume": int(row["volume"]),
                            }
                        )

                    # 從策略對象中提取 LLM 決策記錄
                    llm_decisions = []
                    if hasattr(strategy, "decision_log"):
                        for log_entry in strategy.decision_log:
                            # 安全處理 decision，確保不會對 None 調用 .get()
                            decision = log_entry.get("decision", {})
                            if decision is None:
                                decision = {}

                            llm_decisions.append(
                                {
                                    "timestamp": log_entry["date"].isoformat()
                                    if hasattr(log_entry["date"], "isoformat")
                                    else str(log_entry["date"]),
                                    "decision": decision,
                                    "reasoning": log_entry.get("reasoning", ""),
                                    "events": log_entry.get("events", []),
                                    "action": "THINK",  # LLM 思考決策，非交易信號
                                    "confidence": decision.get("confidence", 0.8),
                                    "price": 0.0,  # 這將在前端根據時間戳查找對應價格
                                }
                            )

                    result_data = {
                        "type": "result",
                        "data": {
                            "trades": backtest_result.get("trades", []),
                            "performance": backtest_result.get(
                                "performance_metrics", {}
                            ),
                            "stock_data": stock_data_json,
                            "signals": backtest_result.get(
                                "trading_signals", []
                            ),  # 修正：使用 trading_signals 而不是 signals
                            "llm_decisions": llm_decisions,
                            "statistics": {
                                # 使用策略統計數據，這些是實際的交易損益數據
                                "total_trades": backtest_result.get(
                                    "strategy_statistics", {}
                                ).get("total_trades", 0),
                                "win_rate": backtest_result.get(
                                    "strategy_statistics", {}
                                ).get("strategy_win_rate", 0)
                                * 100,  # 轉換為百分比
                                "total_return": backtest_result.get(
                                    "strategy_statistics", {}
                                ).get("cumulative_trade_return_rate", 0)
                                * 100,  # 轉換為百分比
                                "max_drawdown": backtest_result.get(
                                    "performance_metrics", {}
                                ).get("max_drawdown", 0),
                                "final_value": backtest_result.get(
                                    "performance_metrics", {}
                                ).get("final_value", 0),
                                "annual_return": backtest_result.get(
                                    "performance_metrics", {}
                                ).get("annual_return", 0),
                                "volatility": backtest_result.get(
                                    "performance_metrics", {}
                                ).get("volatility", 0),
                                "num_trades": backtest_result.get(
                                    "strategy_statistics", {}
                                ).get("total_trades", 0),
                                # 新增策略專用的實現損益
                                "total_realized_pnl": backtest_result.get(
                                    "strategy_statistics", {}
                                ).get("total_realized_pnl", 0),
                                "winning_trades": backtest_result.get(
                                    "strategy_statistics", {}
                                ).get("winning_trades", 0),
                            },
                            "strategy_statistics": backtest_result.get(
                                "strategy_statistics", {}
                            ),
                        },
                    }

                    message_queue.put(result_data)
                    message_queue.put(
                        {"type": "complete", "message": "LLM strategy backtest complete!"}
                    )

                except Exception as e:
                    message_queue.put(
                        {"type": "error", "message": f"An error occurred during the backtest: {str(e)}"}
                    )
                finally:
                    message_queue.put(None)  # End signal

            # Send start signal
            yield f"data: {safe_json_dumps({'type': 'start', 'message': 'Start LLM strategy backtest...'})}\n\n"

            # 在後台線程啟動回測
            backtest_thread = threading.Thread(target=run_backtest)
            backtest_thread.start()

            # 持續從隊列中讀取消息並發送
            while True:
                try:
                    message = message_queue.get(timeout=1)
                    if message is None:  # 結束信號
                        break
                    yield f"data: {safe_json_dumps(message)}\n\n"
                except queue.Empty:
                    # 發送心跳保持連接
                    yield f"data: {safe_json_dumps({'type': 'heartbeat'})}\n\n"
                    continue

            # 等待線程完成
            backtest_thread.join()

        # 返回 SSE 響應
        return StreamingResponse(
            generate_backtest_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            },
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unable to start streaming backtest: {str(e)}")


@router.get("/llm-backtest-stream/status")
async def get_stream_status():
    """
    Check streaming backtest service status
    """
    return {
        "status": "ready",
        "message": "LLM streaming backtest service is running normally",
        "timestamp": datetime.now().isoformat(),
    }
