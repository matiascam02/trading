"""
LLM Smart Strategy
Event-driven intelligent trading strategy based on LLM
"""

import json
import logging
import os
import sqlite3
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from ...utils.backtest_logger import BacktestLogger
from ...utils.indicators import calculate_bollinger_bands, calculate_macd
from ...utils.unrealized_pnl_tracker import UnrealizedPnLTracker
from ..analysis.enhanced_technical_analyzer import EnhancedTechnicalAnalyzer
from ..analysis.trend_analyzer import EnhancedTrendAnalyzer
from ..client import get_llm_client
from ...config import settings
from .base import (
    ParameterSpec,
    ParameterType,
    SignalType,
    StrategyConfig,
    TradingSignal,
    TradingStrategy,
)

logger = logging.getLogger(__name__)


class LLMSmartStrategy(TradingStrategy):
    """
    LLM Smart Strategy

    Workflow:
    1. Analyze stock characteristics with historical data to determine technical indicator parameters
    2. Call LLM for decision making when key events are triggered
    3. 結合趨勢分析優化進出場時機
    """

    def __init__(self, config: StrategyConfig):
        super().__init__(config)
        # Respect explicit provider override if set (e.g., LLM_PROVIDER=openai)
        provider_override = (
            str(settings.LLM_PROVIDER).strip().lower()
            if getattr(settings, "LLM_PROVIDER", None)
            else None
        )
        if provider_override not in {"azure", "openai", "gemini", None}:
            provider_override = None
        self.llm_client = get_llm_client(provider=provider_override, temperature=0.1)
        self.trend_analyzer = EnhancedTrendAnalyzer()
        self.enhanced_analyzer = EnhancedTechnicalAnalyzer()

        # 策略參數
        self.confidence_threshold = config.parameters.get(
            "confidence_threshold", 0.6
        )  # 降低到0.6，增加執行機會
        self.trend_lookback = config.parameters.get("trend_lookback", 20)
        self.event_threshold = config.parameters.get("event_threshold", 0.05)

        # 策略類型選擇 - 預設使用traditional
        self.strategy_type = config.parameters.get("strategy_type", "traditional")

        # 載入決策原則
        self._load_strategy_prompt()
        self.max_daily_trades = config.parameters.get("max_daily_trades", 3)
        self.use_technical_filter = config.parameters.get("use_technical_filter", True)
        self.ma_short = config.parameters.get("ma_short", 10)
        self.ma_long = config.parameters.get("ma_long", 20)

        # 技術指標預設值
        self.macd_fast = 12
        self.macd_slow = 26
        self.macd_signal = 9
        self.analysis_period_months = 3

        # 內部狀態
        self.stock_characteristics = None
        self.current_position = None
        self.decision_log = []
        self.daily_trade_count = 0
        self.last_trade_date = None
        self.current_symbol = None  # 添加當前股票代碼追蹤

        # LLM 呼叫統計
        self.llm_call_count = 0
        self.llm_skipped_count = 0  # 新增：跳過的 LLM 呼叫次數
        self.total_events_detected = 0
        self.events_filtered_out = 0

        # 進度回調函數（用於流式更新）
        self.progress_callback = config.parameters.get("progress_callback", None)

        # 未實現損益追蹤器
        self.pnl_tracker = UnrealizedPnLTracker()
        self.current_position_id = None  # 當前持倉 ID

        # 風險管理參數
        self.max_loss_threshold = config.parameters.get(
            "max_loss_threshold", 0.10
        )  # 10%止損
        self.profit_taking_threshold = config.parameters.get(
            "profit_taking_threshold", 0.15
        )  # 15%獲利了結
        self.position_sizing_adjustment = config.parameters.get(
            "position_sizing_adjustment", True
        )
        self.position_size = config.parameters.get("position_size", 0.2)  # 默認20%倉位

        # LLM 呼叫統計
        self.total_llm_calls = 0
        self.events_filtered_out = 0
        self.total_events_detected = 0

        # 動態績效追蹤
        self.initial_capital = config.parameters.get("initial_capital", 100000)
        self.current_position = None  # 當前持倉狀態
        self.current_symbol = None  # 當前交易的股票代碼
        self.position_entry_price = 0.0  # 進場價格
        self.position_entry_date = None  # 進場日期
        self.shares = 0  # 持股數量
        self.cash = self.initial_capital  # 現金餘額
        self.total_trades = 0
        self.winning_trades = 0
        self.current_portfolio_value = self.initial_capital
        self.max_portfolio_value = self.initial_capital  # 追蹤最高點
        self.max_drawdown = 0.0
        self.total_realized_pnl = 0.0  # 累積實現損益
        self.trade_returns = []  # 記錄每筆交易的收益率 (百分比)

        # 風險控制相關
        self._last_trend_analysis = None  # 儲存最新趨勢分析供風險檢查使用

        # Backtest logger initialization
        self.backtest_logger = None
        if config.parameters.get("enable_logging", True):
            log_path = config.parameters.get(
                "log_path", "backend/data/backtest_logs.db"
            )
            session_id = config.parameters.get("session_id", None)
            self.backtest_logger = BacktestLogger(log_path, session_id)
            logger.info(f"✅ Backtest logger enabled: {log_path}")

    def _load_strategy_prompt(self) -> None:
        """載入策略決策原則"""
        try:
            # 確定當前文件的路徑
            current_dir = os.path.dirname(os.path.abspath(__file__))
            prompt_dir = os.path.join(current_dir, "prompt")

            # 使用traditional策略文件
            file_path = os.path.join(prompt_dir, "traditional_strategy.md")

            # 讀取策略文件
            if os.path.exists(file_path):
                with open(file_path, "r", encoding="utf-8") as f:
                    self.strategy_prompt = f.read()
                logger.info(f"✅ 成功載入traditional策略: {file_path}")
            else:
                logger.warning(f"⚠️ 策略文件不存在: {file_path}，使用默認策略")
                self.strategy_prompt = self._get_default_strategy_prompt()

        except Exception as e:
            logger.error(f"❌ 載入策略文件失敗: {e}，使用默認策略")
            self.strategy_prompt = self._get_default_strategy_prompt()

    def _get_default_strategy_prompt(self) -> str:
        """獲取默認策略提示"""
        return """
# 默認決策原則

## 基本策略
- uptrend: 可考慮進場或持倉
- downtrend: 應該出場
- consolidation: 謹慎觀望

請以JSON格式回應決策：
```json
{
    "action": "BUY" | "SELL" | "HOLD",
    "confidence": 0.0-1.0,
    "reasoning": "決策理由",
    "risk_level": "low" | "medium" | "high",
    "expected_outcome": "預期結果"
}
```
"""

    def _send_progress(
        self,
        day: int,
        total_days: int,
        event_type: str,
        message: str,
        extra_data: dict = None,
    ):
        """發送進度更新的helper方法"""
        if self.progress_callback:
            try:
                if extra_data is not None:
                    self.progress_callback(
                        day, total_days, event_type, message, extra_data
                    )
                else:
                    # 向後兼容：如果callback不支持extra_data參數，則忽略它
                    import inspect

                    sig = inspect.signature(self.progress_callback)
                    if len(sig.parameters) >= 5:
                        self.progress_callback(
                            day, total_days, event_type, message, None
                        )
                    else:
                        self.progress_callback(day, total_days, event_type, message)
            except TypeError:
                # 向後兼容：如果callback不支持5個參數，使用4個參數
                self.progress_callback(day, total_days, event_type, message)

    def set_symbol(self, symbol: str):
        """設置當前分析的股票代碼"""
        self.current_symbol = symbol

    def generate_signals(self, data: pd.DataFrame) -> List[TradingSignal]:
        """
        生成LLM智能交易信號

        Args:
            data: 包含OHLCV數據的DataFrame

        Returns:
            交易信號列表
        """
        signals = []

        # 檢查輸入數據的有效性
        if data is None:
            print("❌ 錯誤: 輸入數據為空 (None)")
            return signals

        if len(data) < 30:  # 降低數據要求
            print(f"⚠️ 數據量不足: {len(data)} < 30，跳過信號生成")
            return signals

        # 初始化P&L追蹤器（如果還沒有）
        if not hasattr(self, "pnl_tracker") or self.pnl_tracker is None:
            try:
                from ...utils.unrealized_pnl_tracker import UnrealizedPnLTracker

                self.pnl_tracker = UnrealizedPnLTracker()
                print(f"📊 P&L追蹤器初始化完成")
            except ImportError as e:
                print(f"⚠️ 無法導入P&L追蹤器: {e}")
                self.pnl_tracker = None

        # 第一步：分析股票特性（使用前期數據）
        self.stock_characteristics = self._analyze_stock_characteristics(data)

        # 根據股票特性動態調整技術指標參數
        self._adjust_technical_parameters()

        # 計算技術指標
        data = self._calculate_all_indicators(data)

        # 分析趨勢 - 添加嚴格的數據長度檢查
        print(f"🔍 準備趨勢分析數據...")

        # 確保有足夠的數據進行分析
        print(f"🔍 數據檢查: 總數據量 = {len(data)}")
        if len(data) < 50:
            print(f"⚠️ 數據量不足進行趨勢分析 ({len(data)} < 50)，使用簡化分析")
            # 創建簡化的趨勢分析結果
            from types import SimpleNamespace

            trend_analysis = SimpleNamespace()
            trend_analysis.dominant_trend = "sideways"
            trend_analysis.complexity_score = 0.5
            trend_analysis.confidence = 0.3
        else:
            print(f"✅ 數據量充足 ({len(data)} >= 50)，開始完整趨勢分析")
            # 將 DataFrame 轉換為所需的格式，並進行數據驗證
            market_data_list = []
            valid_rows = 0

            for idx, row in data.iterrows():
                # 檢查數據完整性
                close_price = row["close"]
                if pd.isna(close_price) or close_price <= 0:
                    print(f"⚠️ 跳過無效數據行: {idx}, close_price={close_price}")
                    continue

                market_data_list.append(
                    {
                        "date": idx.strftime("%Y-%m-%d")
                        if hasattr(idx, "strftime")
                        else str(idx),
                        "close": float(close_price),
                        "open": float(row["open"])
                        if "open" in row and not pd.isna(row["open"])
                        else float(close_price),
                        "high": float(row["high"])
                        if "high" in row and not pd.isna(row["high"])
                        else float(close_price),
                        "low": float(row["low"])
                        if "low" in row and not pd.isna(row["low"])
                        else float(close_price),
                        "volume": float(row["volume"])
                        if "volume" in row and not pd.isna(row["volume"])
                        else 0,
                    }
                )
                valid_rows += 1

            print(f"📊 有效數據行數: {valid_rows}/{len(data)}")

            # 再次檢查清理後的數據量
            if len(market_data_list) < 30:
                print(
                    f"⚠️ 清理後數據量不足 ({len(market_data_list)} < 30)，使用簡化分析"
                )
                from types import SimpleNamespace

                trend_analysis = SimpleNamespace()
                trend_analysis.dominant_trend = "sideways"
                trend_analysis.complexity_score = 0.5
                trend_analysis.confidence = 0.3
            else:
                print(
                    f"✅ 清理後數據量充足 ({len(market_data_list)} >= 30)，調用趨勢分析器..."
                )
                print(f"🔍 開始Enhanced趨勢分析...")
                try:
                    symbol = self.current_symbol or "UNKNOWN"
                    print(f"📈 分析股票: {symbol}")

                    # Get current date from market data
                    current_date = None
                    if market_data_list:
                        last_data = market_data_list[-1]
                        current_date = (
                            last_data.get("date")
                            if isinstance(last_data, dict)
                            else None
                        )
                        print(f"📅 當前日期: {current_date}")

                    # 確保 current_date 是字符串格式
                    current_date_str = None
                    if current_date:
                        if hasattr(current_date, "strftime"):
                            current_date_str = current_date.strftime("%Y-%m-%d")
                        elif isinstance(current_date, str):
                            current_date_str = current_date
                        else:
                            current_date_str = str(current_date)

                    enhanced_result = self.trend_analyzer.analyze_with_llm_optimization(
                        symbol, current_date_str
                    )

                    # Extract traditional trend analysis for compatibility
                    trend_analysis = enhanced_result.original_result

                    # Store enhanced results for later use in prompts
                    self.current_enhanced_analysis = enhanced_result

                    print(f"✅ Enhanced趨勢分析完成: {enhanced_result.market_phase}")
                    print(f"🎯 轉折概率: {enhanced_result.reversal_probability:.2f}")
                    print(f"📊 趨勢一致性: {enhanced_result.trend_consistency:.2f}")
                    print(f"📈 動量狀態: {enhanced_result.momentum_status}")
                    print(f"🔍 主導趨勢: {trend_analysis.dominant_trend}")

                except Exception as e:
                    print(f"❌ Enhanced趨勢分析失敗: {e}")
                    import traceback

                    print(f"🔍 錯誤詳情: {traceback.format_exc()}")
                    # 創建備用分析結果
                    from types import SimpleNamespace

                    trend_analysis = SimpleNamespace()
                    trend_analysis.dominant_trend = "sideways"
                    trend_analysis.complexity_score = 0.5
                    trend_analysis.confidence = 0.2
                    self.current_enhanced_analysis = None

        print(f"🔄 開始事件驅動信號生成 (數據長度: {len(data)})...")
        # 事件驅動的信號生成
        self._total_days = len(data)  # 設置總天數供其他方法使用
        self._last_performance_update_day = -1  # 追蹤上次績效更新的天數，避免重複
        self._last_trend_update_day = -1  # 追蹤上次趨勢更新的天數

        for i in range(30, len(data)):  # 從30天開始，而不是100天
            self._current_day_index = i  # 設置當前索引供其他方法使用

            # 定期重新分析全局趨勢（每30天或重要變化時）
            if i % 30 == 0 and i != self._last_trend_update_day:
                print(f"🔄 第{i}天：重新分析全局趨勢...")
                try:
                    current_data_for_trend = data.iloc[: i + 1].copy()
                    market_data_list = []

                    # 重新構建市場數據
                    for idx, row in current_data_for_trend.iterrows():
                        market_data_list.append(
                            {
                                "date": idx,
                                "open": row.get("open", row.get("Open", 0)),
                                "high": row.get("high", row.get("High", 0)),
                                "low": row.get("low", row.get("Low", 0)),
                                "close": row.get("close", row.get("Close", 0)),
                                "volume": row.get("volume", row.get("Volume", 0)),
                            }
                        )

                    if len(market_data_list) >= 30:
                        symbol = self.current_symbol or "UNKNOWN"
                        current_date = market_data_list[-1].get("date")

                        # 確保 current_date 是字符串格式
                        if hasattr(current_date, "strftime"):
                            current_date_str = current_date.strftime("%Y-%m-%d")
                        elif isinstance(current_date, str):
                            current_date_str = current_date
                        else:
                            current_date_str = str(current_date)

                        enhanced_result = (
                            self.trend_analyzer.analyze_with_llm_optimization(
                                symbol, current_date_str
                            )
                        )
                        trend_analysis = enhanced_result.original_result
                        self.current_enhanced_analysis = enhanced_result

                        print(f"📊 更新全局趨勢: {enhanced_result.market_phase}")
                        print(
                            f"🎯 轉折概率: {enhanced_result.reversal_probability:.2f}"
                        )
                        print(f"📈 動量狀態: {enhanced_result.momentum_status}")

                        self._last_trend_update_day = i

                except Exception as e:
                    print(f"⚠️ 全局趨勢更新失敗: {e}")

            if i % 50 == 0:  # 每50天輸出一次進度
                progress_percentage = (i / len(data) * 100) if len(data) > 0 else 0
                progress_msg = (
                    f"📊 處理進度: {i}/{len(data)} ({progress_percentage:.1f}%)"
                )
                print(progress_msg)

                # 如果有進度回調，發送進度更新
                if self.progress_callback:
                    self.progress_callback(
                        i, len(data), "processing", progress_msg, None
                    )

            # 每10天發送一次績效更新 (包括P&L狀態)，但避免與交易後更新重複
            if (
                self.progress_callback
                and i % 10 == 0
                and i != self._last_performance_update_day
            ):
                current_row = data.iloc[i]
                current_price = current_row.get("close", current_row.get("Close", 0))
                if current_price > 0:
                    self._send_performance_update(i, len(data), current_price)
                    self._last_performance_update_day = i

            current_date = data.index[i]
            historical_data = data.iloc[: i + 1]

            # 安全獲取時間戳 - 處理可能的整數索引
            try:
                if hasattr(current_date, "date"):
                    # 是日期時間對象
                    timestamp = current_date
                    current_date_obj = current_date.date()
                else:
                    # 是整數索引，創建一個默認時間戳
                    timestamp = pd.Timestamp("2024-01-01") + pd.Timedelta(days=i)
                    current_date_obj = i

                if self.last_trade_date != current_date_obj:
                    self.daily_trade_count = 0
                    self.last_trade_date = current_date_obj
            except Exception:
                # 如果日期處理失敗，使用默認值
                timestamp = pd.Timestamp("2024-01-01") + pd.Timedelta(days=i)
                current_date_obj = i
                if self.last_trade_date != current_date_obj:
                    self.daily_trade_count = 0
                    self.last_trade_date = current_date_obj

            if self.daily_trade_count >= self.max_daily_trades:
                continue

            # 檢測觸發事件
            events = self._detect_trigger_events(historical_data, i)
            self.total_events_detected += len(events)

            if events:
                current_price = historical_data.iloc[i]["close"]
                current_date = historical_data.index[i].strftime("%Y-%m-%d")

                # 顯示當前P&L狀態
                if self.current_position:
                    position_metrics = self._calculate_position_metrics(
                        current_price, historical_data.index[i]
                    )
                    unrealized_pnl = position_metrics.get("unrealized_pnl", 0)
                    unrealized_pnl_pct = position_metrics.get("unrealized_pnl_pct", 0)
                    holding_days = position_metrics.get("holding_days", 0)

                    print(f"🎯 第{i}天檢測到事件: {[e['event_type'] for e in events]}")
                    print(
                        f"💰 持倉狀態: {self.shares}股@${self.position_entry_price:.2f}, 現價${current_price:.2f}"
                    )
                    print(
                        f"📊 未實現損益: ${unrealized_pnl:,.2f} ({unrealized_pnl_pct:+.2f}%), 持有{holding_days}天"
                    )
                else:
                    print(f"🎯 第{i}天檢測到事件: {[e['event_type'] for e in events]}")
                    print(
                        f"💵 空倉狀態: 現金${self.cash:,.2f}, 當前價格${current_price:.2f}"
                    )

                # 優化1: 空手 + 震盪市場 + 無強烈信號 才跳過LLM
                if (
                    not self.current_position
                    and trend_analysis.dominant_trend == "sideways"
                ):
                    # 檢查是否有強烈的技術信號
                    strong_signals = [
                        e
                        for e in events
                        if e["event_type"]
                        in [
                            "BB_LOWER_TOUCH",
                            "BB_UPPER_TOUCH",
                            "MACD_GOLDEN_CROSS",
                            "MACD_DEATH_CROSS",
                        ]
                    ]
                    if not strong_signals:
                        print(
                            f"⏭️ 空手+震盪市場+無強烈信號，跳過LLM決策 (趨勢: {trend_analysis.dominant_trend})"
                        )
                        self.events_filtered_out += len(events)
                        continue
                    else:
                        print(
                            f"✅ 震盪市場但有強烈信號 {[s['event_type'] for s in strong_signals]}，繼續LLM決策"
                        )

                # 優化2: 根據持倉狀態篩選相關事件
                relevant_events = self._filter_relevant_events(
                    events, self.current_position
                )
                self.events_filtered_out += len(events) - len(relevant_events)

                if not relevant_events:
                    print(
                        f"⏭️ 無相關事件 (持倉狀態: {'有倉' if self.current_position else '空倉'})，跳過LLM決策"
                    )
                    continue

                # 優化3: 額外檢查 - 持倉時如果是上升趨勢且沒有長黑K棒或其他強烈信號也跳過
                if self.current_position and trend_analysis.dominant_trend == "uptrend":
                    # 檢查是否有長黑K棒或其他強烈出場信號
                    has_large_drop = any(
                        event.get("event_type") == "LARGE_DROP"
                        for event in relevant_events
                    )
                    has_strong_exit_signal = any(
                        event.get("severity") == "high" for event in relevant_events
                    )

                    if not (has_large_drop or has_strong_exit_signal):
                        print(f"⏭️ 持倉+上升趨勢+無長黑K棒或強烈信號，跳過LLM決策")
                        continue

                print(
                    f"📋 相關事件: {[e['event_type'] for e in relevant_events]} (原事件: {len(events)}, 篩選後: {len(relevant_events)})"
                )
                print(
                    f"📈 當前趨勢: {trend_analysis.dominant_trend}, 持倉: {'有倉' if self.current_position else '空倉'}"
                )

                # 根據趨勢類型和持倉狀態決定是否呼叫 LLM
                skip_llm = False
                skip_reason = ""

                has_position = self.current_position is not None

                if has_position:
                    # 持倉狀態：風險管理優先
                    if trend_analysis.dominant_trend == "downtrend":
                        # 持倉 + 下跌趨勢：需要 LLM 分析止損/出場策略
                        skip_llm = False
                        print(f"⚠️  持倉遇下跌趨勢，呼叫 LLM 分析止損策略")
                    elif trend_analysis.dominant_trend == "sideways":
                        # 持倉 + 盤整：需要 LLM 尋找合適出場點
                        skip_llm = False
                        print(f"📊 持倉遇盤整，呼叫 LLM 尋找最佳出場點")
                    elif trend_analysis.dominant_trend == "uptrend":
                        # 持倉 + 上升趨勢：只有遇到長黑K棒才需要LLM判斷
                        has_large_drop = any(
                            event["event_type"] == "LARGE_DROP"
                            for event in relevant_events
                        )
                        if has_large_drop:
                            skip_llm = False
                            print(
                                f"⚠️  持倉+上升趨勢+長黑K棒，呼叫 LLM 分析獲利了結機會"
                            )
                        else:
                            skip_llm = True
                            skip_reason = "持倉+上升趨勢+無長黑K棒，繼續持有"
                else:
                    # 空倉狀態：進場時機選擇
                    if trend_analysis.dominant_trend == "downtrend":
                        # 空倉 + 下跌趨勢：僅在有強烈反轉信號時呼叫 LLM
                        has_reversal_signal = any(
                            event.get("event_type")
                            in ["REVERSAL_PATTERN", "SUPPORT_BOUNCE"]
                            for event in relevant_events
                        )
                        if has_reversal_signal:
                            skip_llm = False
                            print(f"🔄 下跌趨勢中發現反轉信號，呼叫 LLM 分析抄底機會")
                        else:
                            skip_llm = True
                            skip_reason = "空倉+下跌趨勢+無反轉信號，避免逆勢交易"
                    elif trend_analysis.dominant_trend == "sideways":
                        # 空倉 + 盤整：放寬條件，增加更多進場機會
                        has_breakout_signal = any(
                            event.get("event_type")
                            in [
                                "BREAKOUT",
                                "VOLUME_SPIKE",
                                "MOMENTUM_SHIFT",
                                "TREND_TURN_BULLISH",
                                "TREND_TURN_BEARISH",
                            ]
                            for event in relevant_events
                        )
                        has_strong_reversal = any(
                            event.get("event_type")
                            in ["BB_LOWER_TOUCH", "BB_UPPER_TOUCH"]
                            and event.get("severity") in ["high", "very_high"]
                            for event in relevant_events
                        )
                        has_macd_signal = any(
                            event.get("event_type")
                            in ["MACD_GOLDEN_CROSS", "MACD_DEATH_CROSS"]
                            for event in relevant_events
                        )
                        has_ma_signal = any(
                            event.get("event_type")
                            in ["MA_GOLDEN_CROSS", "MA_DEATH_CROSS"]
                            for event in relevant_events
                        )
                        has_multiple_signals = (
                            len(relevant_events) >= 2
                        )  # 多個技術信號同時出現

                        # 放寬條件：任何技術信號都值得LLM分析
                        if (
                            has_breakout_signal
                            or has_strong_reversal
                            or has_macd_signal
                            or has_ma_signal
                            or has_multiple_signals
                        ):
                            skip_llm = False
                            signal_types = [
                                event["event_type"] for event in relevant_events
                            ]
                            print(
                                f"✅ 震盪市場檢測到技術信號 {signal_types}，呼叫LLM分析機會"
                            )
                        else:
                            skip_llm = True
                            skip_reason = "空倉+盤整趨勢，等待明確突破信號"
                    elif trend_analysis.dominant_trend == "uptrend":
                        # 空倉 + 上升趨勢：正常呼叫 LLM 分析進場機會
                        skip_llm = False
                        print(f"🚀 空倉遇上升趨勢，呼叫 LLM 分析進場機會")

                if skip_llm:
                    # 不呼叫 LLM，但記錄事件和原因
                    self.llm_skipped_count += 1  # 增加跳過計數器
                    event_summary = ", ".join(
                        [e["event_type"] for e in relevant_events]
                    )
                    skip_msg = f"⏭️ {timestamp.strftime('%Y-%m-%d')} {skip_reason} (檢測到事件: {event_summary})"
                    print(skip_msg)

                    # 發送跳過進度消息
                    if self.progress_callback:
                        self.progress_callback(
                            i, len(data), "llm_skipped", skip_msg, None
                        )
                    continue

                # 重新分析當前時間點的趨勢 - 使用Enhanced分析
                print(f"🔍 重新分析 {timestamp.strftime('%Y-%m-%d')} 的趨勢...")

                # 優先使用Enhanced分析，fallback到原始分析
                current_enhanced_analysis = None
                try:
                    # 使用Enhanced Trend Analyzer進行當前時間點分析
                    symbol = self.current_symbol or "UNKNOWN"
                    current_date_str = timestamp.strftime("%Y-%m-%d")
                    current_enhanced_analysis = (
                        self.trend_analyzer.analyze_with_llm_optimization(
                            symbol, current_date_str
                        )
                    )
                    self.current_enhanced_analysis = current_enhanced_analysis
                    print(
                        f"✅ Enhanced趨勢分析: {current_enhanced_analysis.market_phase}"
                    )
                    print(f"📊 動量狀態: {current_enhanced_analysis.momentum_status}")
                    print(
                        f"🎯 轉折概率: {current_enhanced_analysis.reversal_probability:.3f}"
                    )

                    # 創建兼容的trend_analysis對象給其他代碼使用
                    current_trend_analysis = current_enhanced_analysis.original_result

                except Exception as e:
                    print(f"⚠️ Enhanced分析失敗，回退到簡化分析: {e}")
                    current_trend_analysis = self._analyze_current_trend(
                        historical_data, timestamp
                    )
                    current_enhanced_analysis = None

                self._last_trend_analysis = current_trend_analysis  # 儲存供風險檢查使用

                if current_enhanced_analysis:
                    print(
                        f"📊 Enhanced趨勢分析: {current_enhanced_analysis.market_phase}"
                    )
                else:
                    print(f"📊 簡化趨勢分析: {current_trend_analysis.dominant_trend}")

                # 調用LLM做決策
                self.llm_call_count += 1  # 增加計數器

                # 發送 LLM 開始決策的進度消息
                if self.progress_callback:
                    llm_start_msg = (
                        f"🤖 {timestamp.strftime('%Y-%m-%d')} 開始LLM分析..."
                    )
                    self.progress_callback(
                        i, len(data), "llm_decision", llm_start_msg, None
                    )

                llm_decision = self._make_llm_decision(
                    historical_data,
                    timestamp,  # 使用處理後的時間戳
                    relevant_events,  # 使用篩選後的事件
                    current_trend_analysis,  # 傳遞兼容的分析結果，但prompt會使用enhanced
                )

                # 發送LLM決策結果
                if llm_decision:
                    action = llm_decision.get("action", "HOLD")
                    confidence = llm_decision.get("confidence", 0)
                    reason = llm_decision.get("reasoning", "無說明")
                    decision_msg = f"🤖 {timestamp.strftime('%Y-%m-%d')} LLM決策: {action} (信心度: {confidence:.2f}) - {reason}"
                    print(decision_msg)

                    if self.progress_callback:
                        self.progress_callback(
                            i, len(data), "llm_decision", decision_msg, None
                        )
                else:
                    print(f"🤖 {timestamp.strftime('%Y-%m-%d')} LLM決策: 無明確建議")

                # 記錄日誌 - 每日分析數據
                if self.backtest_logger:
                    self._log_daily_analysis(
                        timestamp=timestamp,
                        historical_data=historical_data,
                        i=i,
                        events=events,
                        relevant_events=relevant_events,
                        trend_analysis=current_trend_analysis,
                        llm_decision=llm_decision,
                        comprehensive_context=getattr(
                            self, "current_comprehensive_context", None
                        ),
                    )

                if llm_decision and llm_decision.get("action") in ["BUY", "SELL"]:
                    # 檢查信心度閾值
                    confidence = llm_decision.get("confidence", 0)
                    if confidence >= self.confidence_threshold:
                        # 獲取當前價格
                        current_price = historical_data.iloc[-1]["close"]

                        # 使用原始決策
                        enhanced_decision = llm_decision.copy()

                        signal = self._create_signal_from_decision(
                            enhanced_decision,
                            timestamp,  # 使用處理後的時間戳
                            current_price,
                        )
                        if signal:
                            signals.append(signal)
                            self.daily_trade_count += 1
                            signal_msg = f"✅ 生成交易信號: {signal.signal_type} (信心度: {confidence:.2f} >= 門檻: {self.confidence_threshold:.2f})"
                            print(signal_msg)

                            # 記錄交易信號到日誌
                            if self.backtest_logger:
                                self._log_trading_signal(
                                    timestamp, signal, llm_decision
                                )

                            # 計算當前P&L狀態用於前端顯示
                            pnl_data = {}
                            if hasattr(self, "pnl_tracker") and self.pnl_tracker:
                                try:
                                    current_row = data.iloc[i]
                                    current_price = current_row.get(
                                        "close", current_row.get("Close", 0)
                                    )
                                    position_metrics = self._calculate_position_metrics(
                                        current_price, current_date
                                    )
                                    if position_metrics and position_metrics.get(
                                        "has_position"
                                    ):
                                        pnl_data = {
                                            "unrealized_pnl": position_metrics.get(
                                                "unrealized_pnl", 0
                                            ),
                                            "unrealized_pnl_pct": position_metrics.get(
                                                "unrealized_pnl_pct", 0
                                            ),
                                            "holding_days": position_metrics.get(
                                                "holding_days", 0
                                            ),
                                            "shares": position_metrics.get("shares", 0),
                                            "risk_level": position_metrics.get(
                                                "risk_level", "normal"
                                            ),
                                            "cash_remaining": self.cash,
                                            "total_value": self.cash
                                            + (
                                                position_metrics.get("shares", 0)
                                                * current_price
                                            ),
                                        }
                                    else:
                                        pnl_data = {
                                            "unrealized_pnl": 0,
                                            "unrealized_pnl_pct": 0,
                                            "holding_days": 0,
                                            "shares": 0,
                                            "risk_level": "normal",
                                            "cash_remaining": self.cash,
                                            "total_value": self.cash,
                                        }
                                except Exception as e:
                                    print(f"⚠️ P&L計算失敗: {e}")

                            # 發送交易信號生成進度，包含P&L信息
                            if self.progress_callback:
                                extra_data = (
                                    {"pnl_status": pnl_data} if pnl_data else None
                                )
                                self.progress_callback(
                                    i,
                                    len(data),
                                    "signal_generated",
                                    signal_msg,
                                    extra_data,
                                )

                                # 在信號生成後立即發送績效更新
                                current_row = data.iloc[i]
                                current_price = current_row.get(
                                    "close", current_row.get("Close", 0)
                                )
                                if current_price > 0:
                                    self._send_performance_update(
                                        i, len(data), current_price
                                    )
                    else:
                        print(
                            f"❌ 信心度不足: {llm_decision.get('confidence', 0):.2f} < {self.confidence_threshold}"
                        )

        print(f"🎉 信號生成完成! 總共生成 {len(signals)} 個信號")

        # 輸出優化統計
        print(f"")
        print(f"📊 LLM 呼叫優化統計:")
        print(f"   📈 總交易日數: {len(data)} 天")
        print(f"   🎯 總檢測事件: {self.total_events_detected} 個")
        print(f"   🗑️ 篩選掉事件: {self.events_filtered_out} 個")
        print(f"   🤖 LLM 實際呼叫: {self.llm_call_count} 次")
        print(f"   ⏭️  LLM 跳過次數: {self.llm_skipped_count} 次 (下跌/盤整趨勢)")

        # 安全計算效率，避免除零錯誤
        data_length = len(data) if len(data) > 0 else 1
        print(f"   ⚡ 實際呼叫效率: {self.llm_call_count / data_length:.3f} 次/天")

        total_potential_calls = self.llm_call_count + self.llm_skipped_count
        if total_potential_calls > 0:
            print(
                f"   🎯 趨勢過濾率: {self.llm_skipped_count / total_potential_calls:.1%}"
            )
        if self.total_events_detected > 0:
            print(
                f"   🎯 事件處理率: {(self.total_events_detected - self.events_filtered_out) / self.total_events_detected:.1%}"
            )
        print(
            f"   💰 成本節省: {(1 - self.llm_call_count / data_length) * 100:.1f}% (相比每天呼叫)"
        )
        print(
            f"   💡 智能節省: {(1 - self.llm_call_count / (self.llm_call_count + self.llm_skipped_count)) * 100:.1f}% (相比所有事件都呼叫)"
            if total_potential_calls > 0
            else ""
        )

        return signals

    def _analyze_stock_characteristics(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        深度分析股票特性（使用前期數據智能判斷股性）

        Args:
            data: 股票數據

        Returns:
            股票特性分析結果
        """
        # 使用前3-6個月的數據分析，但至少需要60天
        analysis_days = max(60, self.analysis_period_months * 30)
        analysis_data = data.iloc[: min(analysis_days, len(data) // 2)]

        if len(analysis_data) < 30:
            analysis_data = data.iloc[:30] if len(data) >= 30 else data

        print(f"📈 分析股票特性（使用 {len(analysis_data)} 天歷史數據）...")

        # 計算基本統計特性
        returns = analysis_data["close"].pct_change().dropna()
        prices = analysis_data["close"]

        # 確保有足夠的數據進行計算
        if len(returns) < 2:
            logger.warning(
                f"Insufficient data for analysis: only {len(returns)} return values"
            )
            return None

        # 1. 波動性分析（多維度） - 正確的 pandas 語法
        daily_volatility = returns.std() if len(returns) > 1 else 0.0
        annualized_volatility = daily_volatility * np.sqrt(252)

        # 計算波動性的波動性，確保有足夠的滾動窗口數據
        rolling_volatility = returns.rolling(10, min_periods=5).std()
        volatility_of_volatility = (
            rolling_volatility.std() if len(rolling_volatility.dropna()) > 1 else 0.0
        )

        # 2. 趨勢特性分析
        trend_consistency = self._calculate_trend_consistency(analysis_data)
        trend_strength = self._calculate_trend_strength(analysis_data)

        # 3. 價格行為分析 - 安全計算避免除零和 NaN
        price_mean = prices.mean()
        if price_mean > 0:
            price_range_ratio = (prices.max() - prices.min()) / price_mean
        else:
            price_range_ratio = 0.0

        avg_daily_return = returns.mean() if len(returns) > 0 else 0.0
        skewness = (
            returns.skew() if len(returns) > 2 else 0.0
        )  # 偏度：正值表示右偏（上漲多）
        kurtosis = (
            returns.kurtosis() if len(returns) > 3 else 0.0
        )  # 峰度：衡量極端值出現頻率

        # 4. 反轉特性
        reversal_frequency = self._calculate_reversal_frequency(analysis_data)
        consecutive_days = self._calculate_consecutive_move_tendency(returns)

        # 5. 成交量特性 - 安全計算避免 NaN
        volume_volatility = 0.0
        volume_price_correlation = 0.0
        if "volume" in analysis_data.columns and len(analysis_data["volume"]) > 1:
            volume_changes = analysis_data["volume"].pct_change().dropna()
            if len(volume_changes) > 1:
                volume_volatility = volume_changes.std()
                # 安全計算相關性，確保索引對齊
                common_index = returns.index.intersection(volume_changes.index)
                if len(common_index) > 1:
                    aligned_returns = returns.reindex(common_index)
                    aligned_volume = volume_changes.reindex(common_index)
                    volume_price_correlation = aligned_returns.corr(aligned_volume)

        # 6. 技術指標響應性測試
        macd_effectiveness = self._test_macd_effectiveness(analysis_data)
        ma_crossover_effectiveness = self._test_ma_crossover_effectiveness(
            analysis_data
        )
        bb_effectiveness = self._test_bollinger_bands_effectiveness(analysis_data)

        # 7. 支撐阻力分析
        support_resistance_strength = self._analyze_support_resistance(analysis_data)
        breakout_tendency = self._analyze_breakout_tendency(analysis_data)

        # 8. 股性分類
        stock_personality = self._classify_stock_personality(
            annualized_volatility,
            trend_consistency,
            reversal_frequency,
            macd_effectiveness,
        )

        characteristics = {
            # 波動性指標
            "volatility": annualized_volatility,
            "daily_volatility": daily_volatility,
            "volatility_of_volatility": volatility_of_volatility,
            # 收益特性
            "avg_daily_return": avg_daily_return,
            "annualized_return": avg_daily_return * 252,
            "sharpe_ratio": avg_daily_return / daily_volatility
            if daily_volatility > 0
            else 0,
            "skewness": skewness,
            "kurtosis": kurtosis,
            # 趨勢特性
            "trend_consistency": trend_consistency,
            "trend_strength": trend_strength,
            "reversal_frequency": reversal_frequency,
            "consecutive_move_tendency": consecutive_days,
            # 價格行為
            "price_range_ratio": price_range_ratio,
            "breakout_tendency": breakout_tendency,
            # 成交量特性
            "volume_volatility": volume_volatility,
            "volume_price_correlation": volume_price_correlation,
            # 技術指標響應性
            "macd_effectiveness": macd_effectiveness,
            "ma_crossover_effectiveness": ma_crossover_effectiveness,
            "bollinger_effectiveness": bb_effectiveness,
            # 支撐阻力
            "support_resistance_strength": support_resistance_strength,
            # 綜合分類
            "stock_personality": stock_personality,
        }

        # 輸出分析結果
        print(f"📊 股票特性分析完成:")
        print(f"   年化波動率: {annualized_volatility:.1%}")
        print(f"   年化收益率: {characteristics['annualized_return']:.1%}")
        print(f"   夏普比率: {characteristics['sharpe_ratio']:.2f}")
        print(f"   趨勢一致性: {trend_consistency:.2f}")
        print(f"   反轉頻率: {reversal_frequency:.2f}")
        print(f"   股性分類: {stock_personality}")
        print(f"   MACD有效性: {macd_effectiveness:.2f}")

        return characteristics

    def _adjust_technical_parameters(self):
        """根據股票特性智能調整技術指標參數"""
        if not self.stock_characteristics:
            return

        print(f"📊 股票特性分析結果:")
        print(f"   波動性: {self.stock_characteristics['volatility']:.3f}")
        print(f"   趨勢一致性: {self.stock_characteristics['trend_consistency']:.3f}")
        print(f"   反轉頻率: {self.stock_characteristics['reversal_frequency']:.3f}")
        print(
            f"   MACD有效性: {self.stock_characteristics.get('macd_effectiveness', 0.5):.3f}"
        )

        # 保存原始參數作為基準
        original_ma_short = self.ma_short
        original_ma_long = self.ma_long
        original_macd_fast = self.macd_fast
        original_macd_slow = self.macd_slow

        # 1. 根據趨勢一致性調整MACD參數
        trend_consistency = self.stock_characteristics["trend_consistency"]
        print(f"🔍 調整MACD參數 - 趨勢一致性: {trend_consistency:.3f}")
        if trend_consistency > 0.8:  # 趨勢性極強
            self.macd_fast = 6  # 快速捕捉趨勢
            self.macd_slow = 18
            print(f"   趨勢性極強 -> MACD設為 6/18")
        elif trend_consistency > 0.6:  # 趨勢性強
            self.macd_fast = 8
            self.macd_slow = 21
            print(f"   趨勢性強 -> MACD設為 8/21")
        elif trend_consistency > 0.4:  # 中等趨勢性
            self.macd_fast = 12  # 標準設置
            self.macd_slow = 26
            print(f"   中等趨勢性 -> MACD設為 12/26")
        elif trend_consistency > 0.2:  # 趨勢性弱，偏震盪
            self.macd_fast = 15
            self.macd_slow = 35
            print(f"   趨勢性弱 -> MACD設為 15/35")
        else:  # 強震盪性
            self.macd_fast = 20  # 長週期，減少假信號
            self.macd_slow = 45
            print(f"   強震盪性 -> MACD設為 20/45")

        # 3. 根據反轉頻率調整移動平均線參數
        reversal_freq = self.stock_characteristics["reversal_frequency"]
        print(f"🔍 調整均線參數 - 反轉頻率: {reversal_freq:.3f}")
        if reversal_freq > 0.15:  # 高反轉頻率 - 震盪股
            self.ma_short = max(5, self.ma_short - 2)  # 縮短週期
            self.ma_long = max(15, self.ma_long - 5)
            print(f"   高反轉頻率 -> 縮短均線週期")
        elif reversal_freq < 0.05:  # 低反轉頻率 - 趨勢股
            self.ma_short = min(20, self.ma_short + 3)  # 延長週期
            self.ma_long = min(50, self.ma_long + 10)
            print(f"   低反轉頻率 -> 延長均線週期")

        # 4. 根據技術指標有效性進一步微調
        macd_effectiveness = self.stock_characteristics.get("macd_effectiveness", 0.5)

        # 如果MACD效果不佳，使用更保守的參數
        if macd_effectiveness < 0.4:
            self.macd_fast = min(20, int(self.macd_fast * 1.2))
            self.macd_slow = min(50, int(self.macd_slow * 1.1))
            print(f"   MACD效果不佳 -> 保守參數 {self.macd_fast}/{self.macd_slow}")

        # 5. 價格範圍調整 - 考慮股票價格波動幅度
        print(f"\n🔧 技術指標參數智能調整:")
        print(f"   MACD快線: {original_macd_fast} → {self.macd_fast}")
        print(f"   MACD慢線: {original_macd_slow} → {self.macd_slow}")
        print(f"   短期均線: {original_ma_short} → {self.ma_short}")
        print(f"   長期均線: {original_ma_long} → {self.ma_long}")

        # 確保參數的合理性
        self.macd_fast = max(3, min(20, self.macd_fast))
        self.macd_slow = max(10, min(50, self.macd_slow))
        self.ma_short = max(3, min(20, self.ma_short))
        self.ma_long = max(10, min(50, self.ma_long))

        # 確保快線 < 慢線
        if self.macd_fast >= self.macd_slow:
            self.macd_slow = self.macd_fast + 5
        if self.ma_short >= self.ma_long:
            self.ma_long = self.ma_short + 5

        print(f"\n✅ 最終參數（範圍檢查後）:")
        print(f"   MACD: {self.macd_fast}/{self.macd_slow}")
        print(f"   均線: {self.ma_short}/{self.ma_long}")

    def _calculate_position_metrics(
        self, current_price: float, current_date: pd.Timestamp = None
    ) -> Dict[str, Any]:
        """計算當前持倉的詳細指標"""
        if not self.current_position:
            return {
                "has_position": False,
                "unrealized_pnl": 0.0,
                "unrealized_pnl_pct": 0.0,
                "holding_days": 0,
                "shares": 0,
                "position_value": 0.0,
                "risk_level": "normal",
            }

        # 計算未實現損益
        position_value = self.shares * current_price
        unrealized_pnl = position_value - (self.shares * self.position_entry_price)
        unrealized_pnl_pct = (
            unrealized_pnl / (self.shares * self.position_entry_price) * 100
        )

        # 計算持倉天數
        if self.position_entry_date and current_date:
            if isinstance(self.position_entry_date, str):
                entry_date = datetime.strptime(self.position_entry_date, "%Y-%m-%d")
            elif hasattr(self.position_entry_date, "date"):
                entry_date = self.position_entry_date
            else:
                entry_date = pd.to_datetime(self.position_entry_date)

            if hasattr(current_date, "date"):
                current_date_obj = current_date
            else:
                current_date_obj = pd.to_datetime(current_date)

            holding_days = (current_date_obj - entry_date).days
        else:
            holding_days = 0

        # 風險水平評估
        risk_level = self._assess_risk_level(unrealized_pnl_pct, holding_days)

        return {
            "has_position": True,
            "entry_price": self.position_entry_price,
            "current_price": current_price,
            "shares": self.shares,
            "position_value": position_value,
            "unrealized_pnl": unrealized_pnl,
            "unrealized_pnl_pct": unrealized_pnl_pct,
            "holding_days": holding_days,
            "risk_level": risk_level,
            "cost_basis": self.shares * self.position_entry_price,
        }

    def _calculate_current_performance(self, current_price: float) -> Dict[str, float]:
        """計算當前整體績效指標"""
        # 計算當前總價值
        position_value = self.shares * current_price if self.shares > 0 else 0
        total_value = self.cash + position_value

        # 計算總回報率
        total_return = (total_value - self.initial_capital) / self.initial_capital

        # 計算勝率
        win_rate = (
            self.winning_trades / self.total_trades if self.total_trades > 0 else 0
        )

        # 計算累積交易收益率 (每筆交易收益率的總和)
        cumulative_trade_return_rate = sum(self.trade_returns) / 100  # 轉為小數形式

        # 更新最高點和回撤
        if total_value > self.max_portfolio_value:
            self.max_portfolio_value = total_value

        # 計算當前回撤
        current_drawdown = (
            (self.max_portfolio_value - total_value) / self.max_portfolio_value
            if self.max_portfolio_value > 0
            else 0
        )
        if current_drawdown > self.max_drawdown:
            self.max_drawdown = current_drawdown

        return {
            "total_return": total_return,
            "win_rate": win_rate,
            "max_drawdown": self.max_drawdown,
            "total_trades": self.total_trades,
            "total_value": total_value,
            "cash": self.cash,
            "position_value": position_value,
            "total_realized_pnl": self.total_realized_pnl,  # 累積實現損益
            "cumulative_trade_return_rate": cumulative_trade_return_rate,  # 累積交易收益率
        }

    def _send_performance_update(self, day: int, total_days: int, current_price: float):
        """發送績效更新消息"""
        if not self.progress_callback:
            return

        performance = self._calculate_current_performance(current_price)

        # 計算當前P&L狀態
        pnl_status = None
        if hasattr(self, "pnl_tracker") and self.pnl_tracker and current_price > 0:
            try:
                position_metrics = self._calculate_position_metrics(current_price)
                if position_metrics["has_position"]:
                    pnl_status = {
                        "unrealized_pnl": position_metrics["unrealized_pnl"],
                        "unrealized_pnl_pct": position_metrics["unrealized_pnl_pct"],
                        "holding_days": position_metrics["holding_days"],
                        "shares": position_metrics["shares"],
                        "risk_level": position_metrics["risk_level"],
                        "cash_remaining": self.cash,
                        "total_value": self.cash + position_metrics["position_value"],
                    }
                else:
                    # 無持倉時也發送完整的P&L狀態
                    pnl_status = {
                        "unrealized_pnl": 0,
                        "unrealized_pnl_pct": 0,
                        "holding_days": 0,
                        "shares": 0,
                        "risk_level": "normal",
                        "cash_remaining": self.cash,
                        "total_value": self.cash,
                    }
            except Exception as e:
                print(f"⚠️ P&L計算失敗: {e}")

        # 構建與前端期望格式匹配的消息
        message = f"總回報: {performance['total_return'] * 100:+.2f}% | 勝率: {performance['win_rate'] * 100:.1f}% | 最大回撤: {performance['max_drawdown'] * 100:.2f}%"

        # 同時在extra_data中發送詳細數據
        extra_data = {"performance_metrics": performance, "pnl_status": pnl_status}

        self._send_progress(day, total_days, "performance_update", message, extra_data)

    def _assess_risk_level(self, pnl_pct: float, holding_days: int) -> str:
        """評估當前持倉的風險水平"""
        if pnl_pct <= -self.max_loss_threshold * 100:
            return "high_loss"  # 高虧損風險
        elif pnl_pct <= -2:
            return "moderate_loss"  # 中等虧損
        elif pnl_pct >= self.profit_taking_threshold * 100:
            return "high_profit"  # 高收益
        elif pnl_pct >= 8:
            return "moderate_profit"  # 中等收益
        elif holding_days > 30:
            return "long_hold"  # 長期持倉
        else:
            return "normal"  # 正常狀態

    def _generate_pnl_insights(
        self, position_metrics: Dict[str, Any]
    ) -> Dict[str, Any]:
        """基於未實現損益生成投資洞察"""
        if not position_metrics["has_position"]:
            return {
                "pnl_signal": "neutral",
                "risk_warning": None,
                "suggested_action": "可考慮新倉位",
                "position_sizing_factor": 1.0,
            }

        pnl_pct = position_metrics["unrealized_pnl_pct"]
        risk_level = position_metrics["risk_level"]
        holding_days = position_metrics["holding_days"]

        insights = {
            "pnl_signal": "neutral",
            "risk_warning": None,
            "suggested_action": "繼續持有",
            "position_sizing_factor": 1.0,
        }

        # 根據損益狀況給出建議
        if risk_level == "high_loss":
            insights.update(
                {
                    "pnl_signal": "stop_loss",
                    "risk_warning": f"虧損已達{pnl_pct:.1f}%，建議考慮止損",
                    "suggested_action": "立即評估止損",
                    "position_sizing_factor": 0.5,
                }
            )
        elif risk_level == "moderate_loss":
            insights.update(
                {
                    "pnl_signal": "caution",
                    "risk_warning": f"目前虧損{pnl_pct:.1f}%，需謹慎操作",
                    "suggested_action": "謹慎評估後續策略",
                    "position_sizing_factor": 0.7,
                }
            )
        elif risk_level == "high_profit":
            insights.update(
                {
                    "pnl_signal": "take_profit",
                    "risk_warning": None,
                    "suggested_action": f"收益達{pnl_pct:.1f}%，可考慮獲利了結",
                    "position_sizing_factor": 0.8,
                }
            )
        elif risk_level == "moderate_profit":
            insights.update(
                {
                    "pnl_signal": "bullish",
                    "risk_warning": None,
                    "suggested_action": f"收益{pnl_pct:.1f}%，表現良好",
                    "position_sizing_factor": 1.2,
                }
            )
        elif risk_level == "long_hold":
            insights.update(
                {
                    "pnl_signal": "review",
                    "risk_warning": f"持倉已{holding_days}天，建議重新評估",
                    "suggested_action": "檢討持倉策略是否需要調整",
                    "position_sizing_factor": 0.9,
                }
            )

        return insights

    def _update_position_state(
        self, action: str, price: float, quantity: int, date: str
    ):
        """更新持倉狀態"""
        if action == "BUY":
            if self.current_position is None:
                # 新開倉
                self.current_position = "long"
                self.position_entry_price = price
                self.position_entry_date = date
                self.shares = quantity
                self.cash -= quantity * price
                print(f"📈 開倉: {quantity}股 @ ${price:.2f}")
            else:
                # 加倉 (暫時簡化，直接平均成本)
                total_cost = self.shares * self.position_entry_price + quantity * price
                self.shares += quantity
                self.position_entry_price = total_cost / self.shares
                self.cash -= quantity * price
                print(
                    f"📈 加倉: +{quantity}股 @ ${price:.2f}, 平均成本: ${self.position_entry_price:.2f}"
                )

        elif action == "SELL":
            if self.current_position is not None:
                # 計算實現損益
                sell_value = quantity * price
                cost_basis = quantity * self.position_entry_price
                realized_pnl = sell_value - cost_basis
                realized_pnl_pct = (realized_pnl / cost_basis) * 100

                self.cash += sell_value
                self.shares -= quantity

                if self.shares <= 0:
                    # 完全平倉
                    print(
                        f"📉 平倉: {quantity}股 @ ${price:.2f}, 實現損益: ${realized_pnl:,.0f} ({realized_pnl_pct:+.1f}%)"
                    )
                    self.current_position = None
                    self.position_entry_price = 0.0
                    self.position_entry_date = None
                    self.shares = 0
                else:
                    # 部分平倉
                    print(
                        f"📉 減倉: -{quantity}股 @ ${price:.2f}, 實現損益: ${realized_pnl:,.0f} ({realized_pnl_pct:+.1f}%)"
                    )

                # 注意：交易統計在_create_signal_from_decision中更新，這裡不重複更新

    def calculate_position_size(self, price: float) -> int:
        """計算建議倉位大小 - 固定1000股"""
        return 1000

    def _calculate_all_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """計算所有技術指標"""
        data = data.copy()

        # MACD
        macd_data = calculate_macd(
            data,
            short_period=self.macd_fast,
            long_period=self.macd_slow,
            signal_period=self.macd_signal,
        )
        data["macd"] = macd_data["macd"]
        data["macd_signal"] = macd_data["macd_signal"]
        data["macd_histogram"] = macd_data["macd_histogram"]

        # 布林帶
        bb_data = calculate_bollinger_bands(data)
        data["bb_upper"] = bb_data["bb_upper"]
        data["bb_middle"] = bb_data["bb_middle"]
        data["bb_lower"] = bb_data["bb_lower"]

        # 移動平均線
        data[f"ma_{self.ma_short}"] = data["close"].rolling(window=self.ma_short).mean()
        data[f"ma_{self.ma_long}"] = data["close"].rolling(window=self.ma_long).mean()
        data["ma_20"] = data["close"].rolling(window=20).mean()
        data["ma_50"] = data["close"].rolling(window=50).mean()

        return data

    def _analyze_current_trend(
        self, historical_data: pd.DataFrame, current_date
    ) -> Any:
        """
        分析當前時間點的趨勢 - 簡化但有效的實時趨勢分析

        Args:
            historical_data: 歷史數據（到當前時間點為止）
            current_date: 當前日期

        Returns:
            當前時間點的趨勢分析結果
        """
        try:
            # 確保有足夠的數據
            if len(historical_data) < 20:
                print(f"⚠️ 數據不足進行趨勢分析 ({len(historical_data)} < 20)")
                from types import SimpleNamespace

                trend_analysis = SimpleNamespace()
                trend_analysis.dominant_trend = "sideways"
                trend_analysis.complexity_score = 0.5
                trend_analysis.confidence = 0.3
                return trend_analysis

            # 使用多時間框架分析
            data = historical_data.copy()

            # 統一列名（處理大小寫問題）
            column_mapping = {}
            for col in data.columns:
                if col.lower() == "close":
                    column_mapping[col] = "close"
                elif col.lower() == "open":
                    column_mapping[col] = "open"
                elif col.lower() == "high":
                    column_mapping[col] = "high"
                elif col.lower() == "low":
                    column_mapping[col] = "low"
                elif col.lower() == "volume":
                    column_mapping[col] = "volume"

            data = data.rename(columns=column_mapping)

            # 確保有 close 價格數據
            if "close" not in data.columns:
                if "Close" in data.columns:
                    data["close"] = data["Close"]
                else:
                    raise ValueError("找不到價格數據（close/Close列）")

            prices = data["close"]

            # 計算多個時間框架的趨勢
            windows = [5, 10, 20]  # 短期、中期、長期
            trends = []
            trend_strengths = []

            for window in windows:
                if len(prices) >= window + 2:
                    # 使用線性回歸計算趨勢
                    recent_prices = prices.tail(window)
                    x = np.arange(len(recent_prices))
                    y = recent_prices.values

                    if len(y) > 1:
                        slope, _ = np.polyfit(x, y, 1)
                        correlation = np.corrcoef(x, y)[0, 1] if len(y) > 1 else 0

                        # 標準化斜率
                        normalized_slope = slope / recent_prices.mean()

                        # 判斷趨勢方向
                        if abs(normalized_slope) < 0.001:  # 幾乎無趨勢
                            trend_direction = "sideways"
                        elif normalized_slope > 0:
                            trend_direction = "uptrend"
                        else:
                            trend_direction = "downtrend"

                        trends.append(trend_direction)
                        trend_strengths.append(abs(correlation))

            # 確定主導趨勢
            if not trends:
                dominant_trend = "sideways"
                confidence = 0.3
            else:
                # 統計各種趨勢的出現次數和強度
                trend_counts = {"uptrend": 0, "downtrend": 0, "sideways": 0}
                weighted_scores = {"uptrend": 0.0, "downtrend": 0.0, "sideways": 0.0}

                for trend, strength in zip(trends, trend_strengths):
                    trend_counts[trend] += 1
                    weighted_scores[trend] += strength

                # 找出加權分數最高的趨勢
                dominant_trend = max(weighted_scores, key=weighted_scores.get)

                # 計算信心度
                total_strength = sum(trend_strengths)
                if total_strength > 0:
                    confidence = weighted_scores[dominant_trend] / total_strength
                    confidence = min(confidence, 1.0)
                else:
                    confidence = 0.3

            # 檢查價格動量確認趨勢
            trend_reversal_detected = False
            reversal_strength = 0.0

            if len(prices) >= 15:
                # 檢測趨勢轉換信號 - 平衡上升和下降檢測閾值
                short_term_change = (
                    prices.iloc[-5:].mean() - prices.iloc[-10:-5].mean()
                ) / prices.iloc[-10:-5].mean()
                medium_term_change = (
                    prices.iloc[-10:].mean() - prices.iloc[-20:-10].mean()
                ) / prices.iloc[-20:-10].mean()

                print(
                    f"📊 轉換信號計算: 短期變化={short_term_change:.4f} ({short_term_change:.2%}), 中期變化={medium_term_change:.4f} ({medium_term_change:.2%})"
                )

                # 平衡檢測上升和下降轉換，使用相同閾值
                reversal_threshold = 0.02  # 統一使用2%閾值
                counter_threshold = 0.01  # 統一使用1%反向閾值

                if (
                    short_term_change > reversal_threshold
                    and medium_term_change < -counter_threshold
                ):
                    trend_reversal_detected = True
                    reversal_strength = abs(short_term_change)
                    print(
                        f"🔄 檢測到上升轉換信號: 短期變化 {short_term_change:.2%}, 中期變化 {medium_term_change:.2%} -> 轉換強度 {reversal_strength:.2%}"
                    )
                elif (
                    short_term_change < -reversal_threshold
                    and medium_term_change > counter_threshold
                ):
                    trend_reversal_detected = True
                    reversal_strength = abs(short_term_change)
                    print(
                        f"🔄 檢測到下降轉換信號: 短期變化 {short_term_change:.2%}, 中期變化 {medium_term_change:.2%} -> 轉換強度 {reversal_strength:.2%}"
                    )

                # 額外檢測：如果當前趨勢與前一週期趨勢不同
                if len(prices) >= 25:
                    very_recent = prices.iloc[-5:].mean()
                    recent = prices.iloc[-10:-5].mean()
                    older = prices.iloc[-15:-10].mean()
                    much_older = prices.iloc[-25:-15].mean()

                    recent_trend = (very_recent - recent) / recent
                    older_trend = (older - much_older) / much_older

                    # 使用相同閾值檢測雙向趨勢改變
                    trend_change_threshold = 0.02  # 統一閾值
                    counter_trend_threshold = 0.015  # 統一反向閾值

                    if (
                        recent_trend > trend_change_threshold
                        and older_trend < -counter_trend_threshold
                    ):
                        trend_reversal_detected = True
                        reversal_strength = max(reversal_strength, abs(recent_trend))
                        print(
                            f"🔄 檢測到趨勢方向改變(上升): 近期{recent_trend:.2%} vs 早期{older_trend:.2%} -> 轉換強度 {reversal_strength:.2%}"
                        )
                    elif (
                        recent_trend < -trend_change_threshold
                        and older_trend > counter_trend_threshold
                    ):
                        trend_reversal_detected = True
                        reversal_strength = max(reversal_strength, abs(recent_trend))
                        print(
                            f"🔄 檢測到趨勢方向改變(下降): 近期{recent_trend:.2%} vs 早期{older_trend:.2%} -> 轉換強度 {reversal_strength:.2%}"
                        )

            # 價格動量檢查 - 修改為不強制覆蓋趨勢，只作為確認
            momentum_factor = 1.0
            if len(prices) >= 10:
                recent_change = (prices.iloc[-1] - prices.iloc[-10]) / prices.iloc[-10]
                momentum_trend = "uptrend" if recent_change > 0 else "downtrend"

                # 動量確認邏輯 - 改為調整信心度而非強制改變趨勢
                if abs(recent_change) > 0.08:  # 提高閾值到8%，減少誤判
                    if recent_change > 0 and dominant_trend == "uptrend":
                        print(f"🔄 價格動量確認上升趨勢 (變化: {recent_change:.2%})")
                        confidence = min(confidence + 0.15, 1.0)
                        momentum_factor = 1.2
                    elif recent_change < 0 and dominant_trend == "downtrend":
                        print(f"🔄 價格動量確認下降趨勢 (變化: {recent_change:.2%})")
                        confidence = min(confidence + 0.15, 1.0)
                        momentum_factor = 1.2
                    elif abs(recent_change) > 0.12:  # 只有在極強動量時才考慮推翻原趨勢
                        if recent_change > 0 and dominant_trend == "downtrend":
                            print(
                                f"� 極強上升動量推翻下降趨勢 (變化: {recent_change:.2%})"
                            )
                            dominant_trend = "uptrend"
                            confidence = 0.7
                        elif recent_change < 0 and dominant_trend == "uptrend":
                            print(
                                f"� 極強下降動量推翻上升趨勢 (變化: {recent_change:.2%})"
                            )
                            dominant_trend = "downtrend"
                            confidence = 0.7

            # 計算複雜度分數
            unique_trends = len(set(trends))
            complexity_score = unique_trends / len(windows) if windows else 0.5

            # 創建趨勢分析結果
            from types import SimpleNamespace

            trend_analysis = SimpleNamespace()
            trend_analysis.dominant_trend = dominant_trend
            trend_analysis.confidence = confidence
            trend_analysis.complexity_score = complexity_score

            # 添加趨勢轉換信息
            trend_analysis.trend_reversal_detected = trend_reversal_detected
            trend_analysis.reversal_strength = reversal_strength

            print(
                f"🎯 實時趨勢分析: {dominant_trend} (信心: {confidence:.2f}, 複雜度: {complexity_score:.2f})"
            )
            if trend_reversal_detected:
                print(f"⚡ 趨勢轉換檢測: 強度 {reversal_strength:.2%}")

            return trend_analysis

        except Exception as e:
            print(f"❌ 實時趨勢分析失敗: {e}")
            import traceback

            traceback.print_exc()

            # 返回備用結果
            from types import SimpleNamespace

            trend_analysis = SimpleNamespace()
            trend_analysis.dominant_trend = "sideways"
            trend_analysis.complexity_score = 0.5
            trend_analysis.confidence = 0.2
            return trend_analysis

    def _detect_trigger_events(
        self, data: pd.DataFrame, current_index: int
    ) -> List[Dict[str, Any]]:
        """
        檢測觸發事件

        Args:
            data: 歷史數據
            current_index: 當前數據索引

        Returns:
            觸發事件列表
        """
        events = []
        current = data.iloc[current_index]
        prev = data.iloc[current_index - 1] if current_index > 0 else current

        # MACD觸發事件
        if (
            current["macd"] > current["macd_signal"]
            and prev["macd"] <= prev["macd_signal"]
        ):
            events.append(
                {
                    "event_type": "MACD_GOLDEN_CROSS",
                    "severity": "high"
                    if current["macd_histogram"] > 0.01
                    else "medium",
                    "description": "MACD金叉信號",
                    "technical_data": {
                        "indicator": "MACD_GOLDEN_CROSS",
                        "value": None,
                        "threshold": None,
                        "strength": "high"
                        if current["macd_histogram"] > 0.01
                        else "medium",
                    },
                }
            )
        elif (
            current["macd"] < current["macd_signal"]
            and prev["macd"] >= prev["macd_signal"]
        ):
            events.append(
                {
                    "event_type": "MACD_DEATH_CROSS",
                    "severity": "high"
                    if current["macd_histogram"] < -0.01
                    else "medium",
                    "description": "MACD死叉信號",
                    "technical_data": {
                        "indicator": "MACD_DEATH_CROSS",
                        "value": None,
                        "threshold": None,
                        "strength": "high"
                        if current["macd_histogram"] < -0.01
                        else "medium",
                    },
                }
            )

        # 布林帶觸發事件
        if current["close"] <= current["bb_lower"] and prev["close"] > prev["bb_lower"]:
            events.append(
                {
                    "event_type": "BB_LOWER_TOUCH",
                    "severity": "high",
                    "description": "價格觸及布林下軌",
                    "technical_data": {
                        "indicator": "BB_LOWER_TOUCH",
                        "value": None,
                        "threshold": None,
                        "strength": "high",
                    },
                }
            )
        elif (
            current["close"] >= current["bb_upper"] and prev["close"] < prev["bb_upper"]
        ):
            events.append(
                {
                    "event_type": "BB_UPPER_TOUCH",
                    "severity": "high",
                    "description": "價格觸及布林上軌",
                    "technical_data": {
                        "indicator": "BB_UPPER_TOUCH",
                        "value": None,
                        "threshold": None,
                        "strength": "high",
                    },
                }
            )

        # 成交量分析事件
        if len(data) >= 10:
            recent_volume = data["volume"].tail(10).mean()
            if current.get("volume", 0) > recent_volume * 2:
                events.append(
                    {
                        "event_type": "VOLUME_SPIKE",
                        "severity": "medium",
                        "description": f"成交量爆增 ({current.get('volume', 0) / recent_volume:.1f}倍)",
                        "technical_data": {
                            "indicator": "VOLUME_SPIKE",
                            "current_volume": int(current.get("volume", 0)),
                            "avg_volume": int(recent_volume),
                            "ratio": float(current.get("volume", 0) / recent_volume),
                            "strength": "high"
                            if current.get("volume", 0) > recent_volume * 3
                            else "medium",
                        },
                    }
                )

        # 價格突破檢測
        if len(data) >= 20:
            high_20 = data["high"].tail(20).max()
            low_20 = data["low"].tail(20).min()

            if current["close"] > high_20 and prev["close"] <= high_20:
                events.append(
                    {
                        "event_type": "PRICE_BREAKOUT_HIGH",
                        "severity": "high",
                        "description": f"突破20日高點 ({high_20:.2f})",
                        "technical_data": {
                            "indicator": "PRICE_BREAKOUT_HIGH",
                            "breakout_level": float(high_20),
                            "current_price": float(current["close"]),
                            "strength": "high",
                        },
                    }
                )
            elif current["close"] < low_20 and prev["close"] >= low_20:
                events.append(
                    {
                        "event_type": "PRICE_BREAKDOWN_LOW",
                        "severity": "high",
                        "description": f"跌破20日低點 ({low_20:.2f})",
                        "technical_data": {
                            "indicator": "PRICE_BREAKDOWN_LOW",
                            "breakdown_level": float(low_20),
                            "current_price": float(current["close"]),
                            "strength": "high",
                        },
                    }
                )

        # 趨勢轉折事件（使用配置的移動平均線參數）
        ma_short_key = f"ma_{self.ma_short}"
        ma_long_key = f"ma_{self.ma_long}"

        if (
            ma_short_key in current
            and ma_long_key in current
            and ma_short_key in prev
            and ma_long_key in prev
        ):
            if (
                current[ma_short_key] > current[ma_long_key]
                and prev[ma_short_key] <= prev[ma_long_key]
            ):
                events.append(
                    {
                        "event_type": "MA_GOLDEN_CROSS",
                        "severity": "medium",
                        "description": f"短期均線({self.ma_short})上穿長期均線({self.ma_long})",
                        "technical_data": {
                            "indicator": "MA_GOLDEN_CROSS",
                            "ma_short": float(current[ma_short_key]),
                            "ma_long": float(current[ma_long_key]),
                            "strength": "medium",
                        },
                    }
                )
            elif (
                current[ma_short_key] < current[ma_long_key]
                and prev[ma_short_key] >= prev[ma_long_key]
            ):
                events.append(
                    {
                        "event_type": "MA_DEATH_CROSS",
                        "severity": "medium",
                        "description": f"短期均線({self.ma_short})下穿長期均線({self.ma_long})",
                        "technical_data": {
                            "indicator": "MA_DEATH_CROSS",
                            "ma_short": float(current[ma_short_key]),
                            "ma_long": float(current[ma_long_key]),
                            "strength": "medium",
                        },
                    }
                )

        # 長黑K棒檢測（單日跌幅8%以上）
        if prev["close"] > 0:  # 避免除零錯誤
            daily_return = (current["close"] - prev["close"]) / prev["close"]
            if daily_return <= -0.08:  # 下跌8%以上
                events.append(
                    {
                        "event_type": "LARGE_DROP",
                        "severity": "high",
                        "description": f"長黑K棒: 單日跌幅{daily_return * 100:.2f}%",
                        "technical_data": {
                            "indicator": "LARGE_DROP",
                            "daily_return": float(daily_return),
                            "magnitude": float(abs(daily_return)),
                            "strength": "high",
                        },
                    }
                )
            elif daily_return >= 0.08:  # 上漲8%以上
                events.append(
                    {
                        "event_type": "LARGE_GAIN",
                        "severity": "high",
                        "description": f"長紅K棒: 單日漲幅{daily_return * 100:.2f}%",
                        "technical_data": {
                            "indicator": "LARGE_GAIN",
                            "daily_return": float(daily_return),
                            "magnitude": float(daily_return),
                            "strength": "high",
                        },
                    }
                )

        # 保留原有的20/50日均線交叉檢測
        if current["ma_20"] > current["ma_50"] and prev["ma_20"] <= prev["ma_50"]:
            events.append(
                {
                    "event_type": "TREND_TURN_BULLISH",
                    "severity": "medium",
                    "description": "20日均線上穿50日均線",
                    "technical_data": {
                        "indicator": "TREND_TURN_BULLISH",
                        "ma20": float(current["ma_20"]),
                        "ma50": float(current["ma_50"]),
                        "strength": "medium",
                    },
                }
            )
        elif current["ma_20"] < current["ma_50"] and prev["ma_20"] >= prev["ma_50"]:
            events.append(
                {
                    "event_type": "TREND_TURN_BEARISH",
                    "severity": "medium",
                    "description": "20日均線下穿50日均線",
                    "technical_data": {
                        "indicator": "TREND_TURN_BEARISH",
                        "ma20": float(current["ma_20"]),
                        "ma50": float(current["ma_50"]),
                        "strength": "medium",
                    },
                }
            )

        return events

    def _filter_relevant_events(
        self, events: List[Dict[str, Any]], current_position: Optional[str]
    ) -> List[Dict[str, Any]]:
        """
        根據持倉狀態篩選相關事件，減少不必要的 LLM 呼叫

        Args:
            events: 所有檢測到的事件
            current_position: 當前持倉狀態 ('long' 或 None)

        Returns:
            與當前持倉狀態相關的事件列表
        """
        if not events:
            return []

        # 如果沒有持倉，關注買入信號相關事件（放寬篩選）
        if not current_position:
            buy_events = [
                "MACD_GOLDEN_CROSS",
                "RSI_OVERSOLD",
                "BB_LOWER_TOUCH",
                "PRICE_ABOVE_MA20",
                "VOLUME_SPIKE",
                "BULLISH_DIVERGENCE",
                "MA_GOLDEN_CROSS",
                "TREND_TURN_BULLISH",  # 添加更多買入相關事件
            ]
            filtered = [
                event
                for event in events
                if any(buy_event in str(event) for buy_event in buy_events)
            ]
            # 如果沒有篩選到任何事件，保留所有重要事件以防遺漏
            if (
                not filtered and len(events) <= 3
            ):  # 如果事件不多且沒有篩選到，保留原事件
                return events
            return filtered

        # 如果持倉中，關注賣出信號相關事件（放寬篩選）
        else:
            sell_events = [
                "MACD_DEATH_CROSS",
                "RSI_OVERBOUGHT",
                "BB_UPPER_TOUCH",
                "PRICE_BELOW_MA20",
                "BEARISH_DIVERGENCE",
                "VOLUME_DECLINE",
                "MA_DEATH_CROSS",
                "TREND_TURN_BEARISH",
                "LARGE_DROP",  # 添加更多賣出相關事件
            ]
            filtered = [
                event
                for event in events
                if any(sell_event in str(event) for sell_event in sell_events)
            ]
            # 如果沒有篩選到任何事件，保留所有重要事件以防遺漏
            if (
                not filtered and len(events) <= 3
            ):  # 如果事件不多且沒有篩選到，保留原事件
                return events
            return filtered

    def set_current_symbol(self, symbol: str) -> None:
        """設置當前交易的股票代碼"""
        self.current_symbol = symbol
        print(f"📊 設置交易標的: {symbol}")

    def finalize_backtest(
        self, final_price: float, final_timestamp: pd.Timestamp
    ) -> None:
        """
        回測結束時強制結算所有持倉

        Args:
            final_price: 最後一個交易日的收盤價
            final_timestamp: 最後一個交易日的時間戳
        """
        if self.shares > 0 and self.current_position:
            print(f"🏁 回測結束，強制結算持倉...")
            print(f"💰 持倉數量: {self.shares} 股")
            print(f"📈 結算價格: ${final_price:.2f}")

            # 計算實現損益
            sale_value = self.shares * final_price
            cost_basis = (
                self.shares * self.position_entry_price
                if self.position_entry_price > 0
                else 0
            )
            realized_pnl = sale_value - cost_basis
            realized_return = (realized_pnl / cost_basis * 100) if cost_basis > 0 else 0

            print(f"💵 結算金額: ${sale_value:,.0f}")
            if cost_basis > 0:
                print(f"🎯 成本基準: ${cost_basis:,.0f}")
                print(f"📊 實現損益: ${realized_pnl:,.0f} ({realized_return:+.2f}%)")

            # 更新累積實現損益和交易統計
            if cost_basis > 0:
                self.total_realized_pnl += realized_pnl
                self.trade_returns.append(realized_return)  # 記錄這筆交易的收益率
                self.total_trades += 1
                is_winning_trade = realized_pnl > 0
                if is_winning_trade:
                    self.winning_trades += 1

                # 計算當前勝率
                current_win_rate = (
                    (self.winning_trades / self.total_trades * 100)
                    if self.total_trades > 0
                    else 0.0
                )
                print(f"💰 累積實現損益: ${self.total_realized_pnl:,.2f}")
                print(
                    f"📊 交易統計: 第 {self.total_trades} 筆交易完成，勝率 {current_win_rate:.1f}% ({self.winning_trades}/{self.total_trades})"
                )

            # 更新現金餘額
            self.cash += sale_value

            # 計算整體回測統計
            total_return = (
                ((self.cash - self.initial_capital) / self.initial_capital * 100)
                if self.initial_capital > 0
                else 0
            )
            print(f"\n📊 === 完整回測統計 ===")
            print(f"💰 初始資金: ${self.initial_capital:,.2f}")
            print(f"💵 最終資金: ${self.cash:,.2f}")
            print(f"📈 總回報率: {total_return:+.2f}%")
            print(f"🎯 累積實現損益: ${self.total_realized_pnl:,.2f}")
            print(f"📊 總交易次數: {self.total_trades}")
            if self.total_trades > 0:
                print(f"✅ 獲利交易: {self.winning_trades}")
                print(
                    f"📊 整體勝率: {self.winning_trades / self.total_trades * 100:.1f}%"
                )
                print(
                    f"💰 平均每筆損益: ${self.total_realized_pnl / self.total_trades:,.2f}"
                )

            # 清除持倉
            final_shares = self.shares  # 保存股數用於創建信號
            self.shares = 0
            self.current_position = None

            # 創建結算交易記錄
            final_signal = TradingSignal(
                timestamp=final_timestamp,
                signal_type=SignalType.SELL,
                price=final_price,
                confidence=1.0,
                reason="回測結束強制結算",
                metadata={"quantity": final_shares},  # 將 quantity 放入 metadata
            )

            # 如果有P&L追蹤器，更新最終狀態
            if (
                hasattr(self, "pnl_tracker")
                and self.pnl_tracker
                and hasattr(self, "current_position_id")
                and self.current_position_id is not None
            ):
                try:
                    self.pnl_tracker.close_position(
                        self.current_position_id,
                        final_price,
                        final_timestamp.strftime("%Y-%m-%d"),
                    )
                    self.current_position_id = None  # 清除持倉 ID
                    print(f"📊 P&L追蹤器已更新最終狀態")
                except Exception as e:
                    print(f"⚠️ 更新P&L追蹤器失敗: {e}")

            print(f"✅ 持倉結算完成，現金餘額: ${self.cash:,.0f}")

        else:
            print(f"🏁 回測結束，無持倉需要結算")
            print(f"💰 最終現金餘額: ${self.cash:,.0f}")

    def get_final_portfolio_value(self, final_price: float) -> float:
        """
        計算回測結束時的總投資組合價值

        Args:
            final_price: 最後一個交易日的收盤價

        Returns:
            總投資組合價值（現金 + 持倉市值）
        """
        cash_value = self.cash
        position_value = self.shares * final_price if self.shares > 0 else 0
        total_value = cash_value + position_value

        print(f"📊 最終投資組合價值:")
        print(f"   💰 現金: ${cash_value:,.0f}")
        print(
            f"   📈 持倉市值: ${position_value:,.0f} ({self.shares} 股 × ${final_price:.2f})"
        )
        print(f"   🎯 總價值: ${total_value:,.0f}")

        return total_value

        # 定義進場信號（空手時關注）
        entry_signals = {
            "BB_LOWER_TOUCH",  # 觸及布林下軌 - 超賣反彈
            "MACD_GOLDEN_CROSS",  # MACD金叉 - 多頭信號
            "MA_GOLDEN_CROSS",  # 均線金叉 - 多頭信號
            "TREND_TURN_BULLISH",  # 趨勢轉多 - 進場信號
        }

        # 定義出場信號（持倉時關注）
        exit_signals = {
            "BB_UPPER_TOUCH",  # 觸及布林上軌 - 超買回調
            "MACD_DEATH_CROSS",  # MACD死叉 - 空頭信號
            "MA_DEATH_CROSS",  # 均線死叉 - 空頭信號
            "TREND_TURN_BEARISH",  # 趨勢轉空 - 出場信號
            "LARGE_DROP",  # 長黑K棒 - 急跌信號
        }

        relevant_events = []

        # 修改：簡化邏輯，讓所有重要事件都被考慮
        # 這樣可以讓LLM同時考慮進場和出場機會
        print(f"🔍 事件篩選 - 考慮所有重要技術信號")

        for event in events:
            event_type = event["event_type"]

            # 保留所有重要的技術信號
            if event_type in entry_signals or event_type in exit_signals:
                relevant_events.append(event)
                signal_category = (
                    "進場相關" if event_type in entry_signals else "出場相關"
                )
                print(f"   ✅ {signal_category}: {event_type} - {event['description']}")
            else:
                print(f"   ❌ 非關鍵信號: {event_type} - 已過濾")

        return relevant_events

    def _make_llm_decision(
        self,
        data: pd.DataFrame,
        current_date: pd.Timestamp,
        events: List[Dict[str, Any]],
        trend_analysis: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """
        讓LLM做交易決策

        Args:
            data: 歷史數據
            current_date: 當前日期
            events: 觸發事件
            trend_analysis: 趨勢分析結果

        Returns:
            LLM決策結果
        """
        try:
            print(f"🧠 開始LLM決策 (事件數: {len(events)})...")
            self.total_llm_calls += 1  # 增加LLM呼叫計數

            # 準備上下文數據
            current_data = data.iloc[-1]
            recent_data = data.tail(5)
            print(f"📊 準備上下文數據完成")

            # 生成全面的技術分析上下文
            current_date_str = current_date.strftime("%Y-%m-%d")
            comprehensive_context = (
                self.enhanced_analyzer.analyze_comprehensive_context(
                    data, current_date_str, lookback_days=10
                )
            )
            print(f"🔬 全面技術分析完成")

            # 儲存全面技術分析上下文供日誌記錄使用
            self.current_comprehensive_context = comprehensive_context

            print(f"📊 準備進行LLM分析...")

            # 計算持倉指標和P&L洞察
            position_metrics = None
            pnl_insights = None

            if hasattr(self, "pnl_tracker") and self.pnl_tracker:
                # 使用正確的列名 (小寫 'close' 而不是大寫 'Close')
                close_price = current_data.get("close", current_data.get("Close", 0))
                position_metrics = self._calculate_position_metrics(
                    close_price, current_date
                )
                pnl_insights = self._generate_pnl_insights(position_metrics)
                print(
                    f"📈 P&L分析完成: 持倉狀態={position_metrics.get('has_position', False)}"
                )

            # 構建LLM提示詞
            prompt = self._build_decision_prompt(
                current_data,
                recent_data,
                events,
                trend_analysis,
                self.stock_characteristics,
                position_metrics,
                pnl_insights,
                comprehensive_context,  # 添加全面技術分析上下文
            )

            # 檢查 prompt 是否為 None
            if prompt is None:
                print("❌ 錯誤: LLM提示詞構建失敗 (返回 None)")
                return None

            print(f"📝 LLM提示詞構建完成 (長度: {len(prompt)}字元)")

            # 調用LLM
            print(f"🤖 正在呼叫LLM...")
            response = self.llm_client.invoke(prompt)

            # 檢查 LLM 響應是否有效
            if response is None:
                print("❌ 錯誤: LLM響應為空 (response is None)")
                return None

            if not hasattr(response, "content") or response.content is None:
                print("❌ 錯誤: LLM響應內容為空 (response.content is None)")
                return None

            print(f"📡 LLM回應接收完成 (長度: {len(response.content)}字元)")

            # 解析LLM響應
            decision = self._parse_llm_response(response.content)
            print(f"🔍 LLM響應解析完成: {decision}")

            # 記錄決策日誌
            self.decision_log.append(
                {
                    "date": current_date,
                    "events": events,
                    "decision": decision,
                    "reasoning": decision.get("reasoning", "") if decision else "",
                }
            )

            return decision

        except Exception as e:
            print(f"❌ LLM decision error: {e}")
            import traceback

            print(f"🔍 錯誤詳情: {traceback.format_exc()}")
            return None

    def _build_decision_prompt(
        self,
        current_data: pd.Series,
        recent_data: pd.DataFrame,
        events: List[Dict[str, Any]],
        trend_analysis: Dict[str, Any],
        stock_characteristics: Dict[str, Any],
        position_metrics: Optional[Dict[str, Any]] = None,
        pnl_insights: Optional[Dict[str, Any]] = None,
        comprehensive_context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """構建LLM決策提示詞"""

        prompt = f"""
你是一個專業的股票交易策略分析師。請基於以下信息做出交易決策：

## 股票特性分析
- 波動性: {stock_characteristics.get("volatility", 0):.3f}
- 趨勢一致性: {stock_characteristics.get("trend_consistency", 0):.3f}
- MACD有效性: {stock_characteristics.get("macd_effectiveness", 0):.3f}

## 當前市場數據
- 當前價格: {current_data["close"]:.2f}
- MACD: {current_data.get("macd", 0):.4f}
- MACD信號線: {current_data.get("macd_signal", 0):.4f}
- 布林上軌: {current_data.get("bb_upper", 0):.2f}
- 布林中軌: {current_data.get("bb_middle", 0):.2f}
- 布林下軌: {current_data.get("bb_lower", 0):.2f}
- {self.ma_short}日均線: {current_data.get(f"ma_{self.ma_short}", 0):.2f}
- {self.ma_long}日均線: {current_data.get(f"ma_{self.ma_long}", 0):.2f}
- 20日均線: {current_data.get("ma_20", 0):.2f}
- 50日均線: {current_data.get("ma_50", 0):.2f}

## 觸發事件
"""

        for event in events:
            prompt += f"- {event['event_type']}: {event['description']} (嚴重性: {event['severity']})\n"

        # 添加全面的技術分析上下文
        if comprehensive_context and not comprehensive_context.get("error"):
            prompt += f"""
## 📊 全面技術分析

### 💰 價格行為分析
- 價格變化: {comprehensive_context.get("price_action", {}).get("price_change_pct", 0):.2f}%
- K線型態: {comprehensive_context.get("price_action", {}).get("candle_type", "unknown")}
- 實體比例: {comprehensive_context.get("price_action", {}).get("body_ratio", 0):.2f}
- 成交量比值: {comprehensive_context.get("price_action", {}).get("volume_to_avg_ratio", 1):.2f}倍
- 跳空: {comprehensive_context.get("price_action", {}).get("gap_pct", 0):.2f}%

### 📈 移動平均線分析
- MA5: ${comprehensive_context.get("moving_averages", {}).get("ma_5", 0):.2f} (斜率: {comprehensive_context.get("moving_averages", {}).get("ma_5_slope", 0):.4f})
- MA10: ${comprehensive_context.get("moving_averages", {}).get("ma_10", 0):.2f} (斜率: {comprehensive_context.get("moving_averages", {}).get("ma_10_slope", 0):.4f})
- MA20: ${comprehensive_context.get("moving_averages", {}).get("ma_20", 0):.2f} (斜率: {comprehensive_context.get("moving_averages", {}).get("ma_20_slope", 0):.4f})
- 均線排列: {comprehensive_context.get("moving_averages", {}).get("ma_alignment", "unknown")}
- 位於所有均線之上: {comprehensive_context.get("moving_averages", {}).get("above_all_mas", False)}

### 📊 成交量分析
- 當前成交量: {comprehensive_context.get("volume_analysis", {}).get("current_volume", 0):,}
- 成交量比值: {comprehensive_context.get("volume_analysis", {}).get("volume_ratio", 1):.2f}倍
- 成交量趨勢: {comprehensive_context.get("volume_analysis", {}).get("volume_trend", 0):.2f}
- 是否爆量: {comprehensive_context.get("volume_analysis", {}).get("is_high_volume", False)}
- 價量配合: {comprehensive_context.get("volume_analysis", {}).get("volume_confirmation", False)}

### 🌊 波動性分析
- ATR: {comprehensive_context.get("volatility_analysis", {}).get("atr", 0):.2f}
- 年化波動率: {comprehensive_context.get("volatility_analysis", {}).get("volatility_annualized", 0):.2f}%
- 波動率百分位: {comprehensive_context.get("volatility_analysis", {}).get("volatility_percentile", 50):.1f}%
- 高波動: {comprehensive_context.get("volatility_analysis", {}).get("is_high_volatility", False)}

### ⚡ 動量指標
- RSI: {comprehensive_context.get("momentum_indicators", {}).get("rsi", 50):.2f}
- RSI狀態: {comprehensive_context.get("momentum_indicators", {}).get("rsi_condition", "neutral")}
- 5日ROC: {comprehensive_context.get("momentum_indicators", {}).get("roc_5_day", 0):.2f}%
- 10日ROC: {comprehensive_context.get("momentum_indicators", {}).get("roc_10_day", 0):.2f}%
- 動量強度: {comprehensive_context.get("momentum_indicators", {}).get("momentum_strength", "neutral")}

### 🎯 支撐阻力
- 最近阻力: ${comprehensive_context.get("support_resistance", {}).get("nearest_resistance", 0):.2f}
- 最近支撐: ${comprehensive_context.get("support_resistance", {}).get("nearest_support", 0):.2f}
- 距阻力: {comprehensive_context.get("support_resistance", {}).get("resistance_distance_pct", 0):.2f}%
- 距支撐: {comprehensive_context.get("support_resistance", {}).get("support_distance_pct", 0):.2f}%
- 接近關鍵位: {comprehensive_context.get("support_resistance", {}).get("near_resistance", False) or comprehensive_context.get("support_resistance", {}).get("near_support", False)}

### 📐 趨勢強度分析
- 趨勢方向: {comprehensive_context.get("trend_analysis", {}).get("trend_direction", "neutral")}
- 趨勢強度: {comprehensive_context.get("trend_analysis", {}).get("trend_strength", 0):.3f}
- ADX值: {comprehensive_context.get("trend_analysis", {}).get("adx_value", 0):.2f}
- 強勢趨勢: {comprehensive_context.get("trend_analysis", {}).get("strong_trend", False)}

### 🏮 市場狀態
- 市場型態: {comprehensive_context.get("market_regime", {}).get("market_regime", "unknown")}
- 型態描述: {comprehensive_context.get("market_regime", {}).get("regime_description", "Unknown regime")}
- 是否趨勢行情: {comprehensive_context.get("market_regime", {}).get("is_trending", False)}
- 是否高波動: {comprehensive_context.get("market_regime", {}).get("is_volatile", False)}

### 🎈 布林通道分析
- 布林位置: {comprehensive_context.get("bollinger_analysis", {}).get("bb_position", 0.5):.3f} (0=下軌, 1=上軌)
- 通道寬度: {comprehensive_context.get("bollinger_analysis", {}).get("bb_width", 0):.2f}%
- 通道收縮: {comprehensive_context.get("bollinger_analysis", {}).get("is_squeeze", False)}
- 潛在突破: {comprehensive_context.get("bollinger_analysis", {}).get("potential_breakout", False)}

### 📈 MACD分析
- MACD線: {comprehensive_context.get("macd_analysis", {}).get("macd_line", 0):.4f}
- 信號線: {comprehensive_context.get("macd_analysis", {}).get("signal_line", 0):.4f}
- 柱狀圖: {comprehensive_context.get("macd_analysis", {}).get("histogram", 0):.4f}
- MACD位置: {comprehensive_context.get("macd_analysis", {}).get("macd_position", "neutral")}
- 交叉信號: {comprehensive_context.get("macd_analysis", {}).get("macd_cross", "none")}
"""

        prompt += f"""
## 趨勢分析"""

        # Use Enhanced analysis if available, otherwise fallback to original
        if (
            hasattr(self, "current_enhanced_analysis")
            and self.current_enhanced_analysis
        ):
            enhanced = self.current_enhanced_analysis
            prompt += f"""
- 主導趨勢: {enhanced.market_phase} (Enhanced分析)
- 趨勢一致性: {enhanced.trend_consistency:.3f}
- 轉折概率: {enhanced.reversal_probability:.3f}
- 動量狀態: {enhanced.momentum_status}
- 風險水平: {enhanced.risk_level}

📊 **趨勢判斷說明**: 
- 使用Enhanced多時間框架分析，market_phase為主要趨勢判斷依據
- {enhanced.market_phase}代表當前主導市場方向
- 趨勢一致性{enhanced.trend_consistency:.3f}表示多時間框架的趨勢統一程度"""
        else:
            # Fallback to original analysis
            prompt += f"""
- 主導趨勢: {trend_analysis.dominant_trend if trend_analysis else "unknown"} (基礎分析)
- 趨勢強度: {trend_analysis.complexity_score if trend_analysis else 0:.3f}

📊 **趨勢判斷說明**: 
- 使用基礎趨勢分析，dominant_trend為主要趨勢判斷依據"""

            # 添加趨勢轉換信息
            if (
                hasattr(trend_analysis, "trend_reversal_detected")
                and trend_analysis.trend_reversal_detected
            ):
                # 根據轉換強度給出重要性評級
                if trend_analysis.reversal_strength > 0.05:  # 5%以上
                    importance = "🔥 強烈轉換信號"
                elif trend_analysis.reversal_strength > 0.03:  # 3%以上
                    importance = "⚡ 明顯轉換信號"
                else:
                    importance = "📊 輕微轉換信號"

                prompt += f"""
- {importance}: 檢測到趨勢轉換點 (強度: {trend_analysis.reversal_strength:.2%})
- 🎯 關鍵時機: 這是潛在的趨勢轉換點，歷史上這類信號往往預示重要機會
- 💡 策略提醒: 轉換信號強度 ≥ 2% 時應該積極考慮進場，≥ 3% 時應該果斷行動"""

        prompt += f"""

## 當前持倉狀態
持倉狀態: {"有持倉" if self.current_position else "空倉"}"""

        # 添加未實現損益信息
        if position_metrics and position_metrics.get("has_position"):
            prompt += f"""

### 📈 持倉詳情
- 持倉數量: {position_metrics["shares"]:,.0f} 股
- 進場價格: ${position_metrics["entry_price"]:.2f}
- 當前價格: ${position_metrics["current_price"]:.2f}
- 持倉成本: ${position_metrics["cost_basis"]:,.0f}
- 當前市值: ${position_metrics["position_value"]:,.0f}

### 💰 未實現損益分析
- 未實現損益: ${position_metrics["unrealized_pnl"]:,.0f}
- 收益率: {position_metrics["unrealized_pnl_pct"]:+.2f}%
- 持倉天數: {position_metrics["holding_days"]} 天
- 風險水平: {position_metrics["risk_level"]}

### 🎯 損益洞察
- 損益信號: {pnl_insights.get("pnl_signal", "neutral") if pnl_insights else "neutral"}
- 風險提醒: {pnl_insights.get("risk_warning", "無特殊風險") if pnl_insights else "無特殊風險"}
- 建議動作: {pnl_insights.get("suggested_action", "正常操作") if pnl_insights else "正常操作"}"""
        else:
            prompt += f"""

### 📈 持倉詳情
- 持倉狀態: 空倉
- 可用資金: ${self.cash:,.0f}
- 總資產: ${self.cash:,.0f}

### 🎯 投資洞察
- 建議動作: {pnl_insights.get("suggested_action", "可考慮新倉位") if pnl_insights else "可考慮新倉位"}
- 倉位建議: 正常倉位配置"""

        # 添加動態載入的策略決策原則
        prompt += f"""

{self.strategy_prompt}
"""

        return prompt

    def _parse_llm_response(self, response: str) -> Optional[Dict[str, Any]]:
        """解析LLM響應"""
        try:
            # 嘗試提取JSON部分
            json_start = response.find("{")
            json_end = response.rfind("}") + 1

            if json_start != -1 and json_end > json_start:
                json_str = response[json_start:json_end]
                return json.loads(json_str)

            return None

        except Exception as e:
            print(f"Error parsing LLM response: {e}")
            return None

    def _create_signal_from_decision(
        self, decision: Dict[str, Any], timestamp: pd.Timestamp, price: float
    ) -> Optional[TradingSignal]:
        """從LLM決策創建交易信號"""

        action = decision.get("action")
        if action not in ["BUY", "SELL"]:
            return None

        # 額外風險檢查：阻止明顯不利的進場
        if (
            action == "BUY"
            and hasattr(self, "_last_trend_analysis")
            and self._last_trend_analysis
        ):
            trend_analysis = self._last_trend_analysis

            # 檢查1: 強烈下跌趨勢中不進場
            if (
                trend_analysis.dominant_trend == "downtrend"
                and hasattr(trend_analysis, "trend_strength")
                and trend_analysis.trend_strength >= 0.8
            ):
                print(
                    f"🚫 風險控制：拒絕在強烈下跌趨勢中進場 (趨勢強度: {trend_analysis.trend_strength:.3f})"
                )
                return None

            # 檢查2: 下跌趨勢且趨勢一致性低時不進場
            if (
                trend_analysis.dominant_trend == "downtrend"
                and hasattr(trend_analysis, "trend_consistency")
                and trend_analysis.trend_consistency < 0.5
            ):
                print(
                    f"🚫 風險控制：下跌趨勢且趨勢不明朗時避免進場 (一致性: {trend_analysis.trend_consistency:.3f})"
                )
                return None

            # 檢查3: 下跌趨勢且強度超過0.5時，需要額外確認
            if (
                trend_analysis.dominant_trend == "downtrend"
                and hasattr(trend_analysis, "trend_strength")
                and trend_analysis.trend_strength >= 0.5
            ):
                print(
                    f"⚠️ 風險警告：在中等強度下跌趨勢中進場，需要高度謹慎 (趨勢強度: {trend_analysis.trend_strength:.3f})"
                )
                # 降低信心度
                decision["confidence"] = min(decision.get("confidence", 0.5), 0.75)

        signal_type = SignalType.BUY if action == "BUY" else SignalType.SELL
        confidence = decision.get("confidence", 0.5)
        reasoning = decision.get("reasoning", "")

        # 更新持倉狀態
        if hasattr(self, "pnl_tracker") and self.pnl_tracker:
            try:
                if action == "BUY" and not self.current_position:
                    # 固定1000股
                    shares_to_buy = 1000
                    cost = shares_to_buy * price

                    print(f"🎯 固定倉位: 買入 {shares_to_buy} 股")

                    # 確保有足夠現金
                    if cost <= self.cash:
                        # 添加新持倉到P&L追蹤器
                        if self.current_symbol:
                            self.current_position_id = self.pnl_tracker.add_position(
                                self.current_symbol,
                                timestamp.strftime("%Y-%m-%d"),
                                price,
                                shares_to_buy,
                                confidence,
                            )

                        # 更新內部持倉狀態
                        self.current_position = "long"
                        self.position_entry_price = price
                        self.position_entry_date = timestamp
                        self.shares = shares_to_buy
                        self.cash -= cost

                        # 使用固定止損比例 (5%)
                        stop_loss_price = price * 0.95

                        print(
                            f"📈 持倉更新: 買入 {shares_to_buy} 股，價格 ${price:.2f}，總成本 ${cost:,.0f}"
                        )
                        print(f"🛡️ 止損設定: ${stop_loss_price:.2f} (5%止損)")

                        # 立即發送交易後的P&L更新
                        if self.progress_callback:
                            try:
                                # 計算當前索引（假設這是在循環中調用的）
                                day_index = getattr(self, "_current_day_index", 0)
                                total_days = getattr(self, "_total_days", 125)
                                self._send_performance_update(
                                    day_index, total_days, price
                                )
                            except Exception as e:
                                print(f"⚠️ 買入後P&L更新失敗: {e}")
                    else:
                        print(
                            f"⚠️ 現金不足，無法買入{shares_to_buy}股 (需要 ${cost:,.0f}，現有 ${self.cash:,.0f})"
                        )
                        return None  # 資金不足時不產生信號

                elif action == "SELL" and self.current_position and self.shares > 0:
                    # 賣出所有持股
                    proceeds = self.shares * price

                    # 計算並記錄實現損益
                    cost_basis = self.shares * self.position_entry_price
                    realized_pnl = proceeds - cost_basis
                    realized_return = (
                        (realized_pnl / cost_basis * 100) if cost_basis > 0 else 0
                    )

                    # 更新累積實現損益
                    self.total_realized_pnl += realized_pnl
                    self.trade_returns.append(realized_return)  # 記錄這筆交易的收益率

                    # 更新交易統計 (一個完整交易：買入 -> 賣出)
                    self.total_trades += 1
                    is_winning_trade = realized_pnl > 0
                    if is_winning_trade:
                        self.winning_trades += 1

                    # 計算當前勝率
                    current_win_rate = (
                        (self.winning_trades / self.total_trades * 100)
                        if self.total_trades > 0
                        else 0.0
                    )

                    # 重置持倉狀態
                    self.current_position = None
                    self.position_entry_price = 0.0
                    self.position_entry_date = None
                    old_shares = self.shares
                    self.shares = 0
                    self.cash += proceeds

                    # 關閉P&L追蹤器中的持倉
                    if (
                        hasattr(self, "pnl_tracker")
                        and self.pnl_tracker
                        and hasattr(self, "current_position_id")
                        and self.current_position_id is not None
                    ):
                        try:
                            self.pnl_tracker.close_position(
                                self.current_position_id,
                                price,
                                timestamp.strftime("%Y-%m-%d"),
                            )
                            self.current_position_id = None  # 清除持倉 ID
                        except Exception as e:
                            print(f"⚠️ P&L追蹤器關閉持倉失敗: {e}")

                    print(f"📉 持倉清空: 賣出 {old_shares} 股，價格 ${price:.2f}")
                    print(f"💰 賣出金額: ${proceeds:,.2f}")
                    print(f"🎯 成本基準: ${cost_basis:,.2f}")
                    print(
                        f"📊 實現損益: ${realized_pnl:,.2f} ({realized_return:+.2f}%) ({'✅ 獲利' if is_winning_trade else '❌ 虧損'})"
                    )
                    print(f"💰 累積實現損益: ${self.total_realized_pnl:,.2f}")
                    print(
                        f"📊 交易統計: 第 {self.total_trades} 筆交易完成，勝率 {current_win_rate:.1f}% ({self.winning_trades}/{self.total_trades})"
                    )
                    print(f"💵 當前現金餘額: ${self.cash:,.2f}")

                    # 立即發送交易後的P&L更新
                    if self.progress_callback:
                        try:
                            # 計算當前索引（假設這是在循環中調用的）
                            day_index = getattr(self, "_current_day_index", 0)
                            total_days = getattr(self, "_total_days", 125)
                            self._send_performance_update(day_index, total_days, price)
                        except Exception as e:
                            print(f"⚠️ 賣出後P&L更新失敗: {e}")

            except Exception as e:
                print(f"⚠️ 持倉狀態更新失敗: {e}")

        return TradingSignal(
            timestamp=timestamp,
            signal_type=signal_type,
            confidence=confidence,
            price=price,
            reason=f"LLM決策: {reasoning}",
            metadata={
                "decision": decision,
                "risk_level": decision.get("risk_level", "medium"),
                "expected_outcome": decision.get("expected_outcome", ""),
                "position_size": getattr(self, "shares", 0),
                "cash_remaining": getattr(self, "cash", 0),
            },
        )

    # 輔助方法（計算股票特性）
    def _calculate_trend_consistency(self, data: pd.DataFrame) -> float:
        """計算趨勢一致性"""
        try:
            returns = data["close"].pct_change().dropna()
            if len(returns) == 0:
                return 0.0

            # 計算連續同向變動的比例
            direction_changes = (returns > 0).astype(int).diff().abs().sum()
            if len(returns) == 0:
                return 0.0

            consistency = 1.0 - (direction_changes / len(returns))
            return max(0.0, min(1.0, consistency))

        except Exception as e:
            print(f"趨勢一致性計算錯誤: {e}")
            return 0.0

    def _calculate_trend_strength(self, data: pd.DataFrame) -> float:
        """計算趨勢強度（基於價格走勢的線性回歸）"""
        if len(data) < 10:
            return 0.0

        try:
            prices = data["close"].dropna()
            if len(prices) < 2:
                return 0.0

            x = np.arange(len(prices))

            # 計算線性回歸的R²值作為趨勢強度指標
            correlation_matrix = np.corrcoef(x, prices)
            if correlation_matrix.size == 0:
                return 0.0

            correlation = abs(correlation_matrix[0, 1])
            if np.isnan(correlation):
                return 0.0

            return correlation**2  # R²值

        except Exception as e:
            print(f"趨勢強度計算錯誤: {e}")
            return 0.0

    def _calculate_consecutive_move_tendency(self, returns: pd.Series) -> float:
        """計算連續移動傾向（動量特性）"""
        if len(returns) < 5:
            return 0.0

        consecutive_up = 0
        consecutive_down = 0
        max_consecutive_up = 0
        max_consecutive_down = 0

        for ret in returns:
            if ret > 0:
                consecutive_up += 1
                consecutive_down = 0
                max_consecutive_up = max(max_consecutive_up, consecutive_up)
            elif ret < 0:
                consecutive_down += 1
                consecutive_up = 0
                max_consecutive_down = max(max_consecutive_down, consecutive_down)
            else:
                consecutive_up = 0
                consecutive_down = 0

        return (
            (max_consecutive_up + max_consecutive_down) / len(returns)
            if len(returns) > 0
            else 0.0
        )

    def _test_ma_crossover_effectiveness(self, data: pd.DataFrame) -> float:
        """測試移動平均線交叉有效性"""
        if len(data) < 50:
            return 0.5

        # 使用10日和20日均線測試
        ma_short = data["close"].rolling(10).mean()
        ma_long = data["close"].rolling(20).mean()

        successful_signals = 0
        total_signals = 0

        for i in range(21, len(data) - 5):
            # 金叉
            if (
                ma_short.iloc[i] > ma_long.iloc[i]
                and ma_short.iloc[i - 1] <= ma_long.iloc[i - 1]
            ):
                future_returns = data["close"].iloc[i + 1 : i + 6].pct_change().sum()
                if future_returns > 0:
                    successful_signals += 1
                total_signals += 1
            # 死叉
            elif (
                ma_short.iloc[i] < ma_long.iloc[i]
                and ma_short.iloc[i - 1] >= ma_long.iloc[i - 1]
            ):
                future_returns = data["close"].iloc[i + 1 : i + 6].pct_change().sum()
                if future_returns < 0:
                    successful_signals += 1
                total_signals += 1

        return successful_signals / total_signals if total_signals > 0 else 0.5

    def _test_bollinger_bands_effectiveness(self, data: pd.DataFrame) -> float:
        """測試布林帶有效性"""
        if len(data) < 40:
            return 0.5

        # 計算布林帶
        bb_data = calculate_bollinger_bands(data, window=20, num_std_dev=2)

        successful_signals = 0
        total_signals = 0

        for i in range(21, len(data) - 5):
            current_price = data["close"].iloc[i]
            bb_upper = bb_data["bb_upper"].iloc[i]
            bb_lower = bb_data["bb_lower"].iloc[i]

            # 觸及下軌（超賣）
            if current_price <= bb_lower:
                future_returns = data["close"].iloc[i + 1 : i + 6].pct_change().sum()
                if future_returns > 0:
                    successful_signals += 1
                total_signals += 1
            # 觸及上軌（超買）
            elif current_price >= bb_upper:
                future_returns = data["close"].iloc[i + 1 : i + 6].pct_change().sum()
                if future_returns < 0:
                    successful_signals += 1
                total_signals += 1

        return successful_signals / total_signals if total_signals > 0 else 0.5

    def _analyze_breakout_tendency(self, data: pd.DataFrame) -> float:
        """分析突破傾向"""
        if len(data) < 20:
            return 0.5

        breakouts = 0
        total_opportunities = 0

        # 使用20日高低點作為突破參考
        rolling_high = data["high"].rolling(20).max()
        rolling_low = data["low"].rolling(20).min()

        for i in range(20, len(data) - 1):
            if data["close"].iloc[i] > rolling_high.iloc[i - 1]:  # 向上突破
                if data["close"].iloc[i + 1] > data["close"].iloc[i]:  # 次日繼續上漲
                    breakouts += 1
                total_opportunities += 1
            elif data["close"].iloc[i] < rolling_low.iloc[i - 1]:  # 向下突破
                if data["close"].iloc[i + 1] < data["close"].iloc[i]:  # 次日繼續下跌
                    breakouts += 1
                total_opportunities += 1

        return breakouts / total_opportunities if total_opportunities > 0 else 0.5

    def _classify_stock_personality(
        self,
        volatility: float,
        trend_consistency: float,
        reversal_frequency: float,
        macd_effectiveness: float,
    ) -> str:
        """基於特性分析結果分類股票性格"""

        if volatility > 0.4 and reversal_frequency > 0.1:
            return "高波動震盪型"
        elif volatility > 0.4 and trend_consistency > 0.6:
            return "高波動趨勢型"
        elif volatility < 0.2 and trend_consistency > 0.7:
            return "穩健趨勢型"
        elif volatility < 0.2 and reversal_frequency > 0.08:
            return "低波動震盪型"
        elif trend_consistency > 0.8:
            return "強趨勢型"
        elif reversal_frequency > 0.12:
            return "高頻反轉型"
        elif macd_effectiveness > 0.7:
            return "技術指標敏感型"
        elif 0.2 <= volatility <= 0.35 and 0.4 <= trend_consistency <= 0.7:
            return "平衡型"
        else:
            return "複雜混合型"

    def _calculate_reversal_frequency(self, data: pd.DataFrame) -> float:
        """計算反轉頻率"""
        if len(data) < 10:
            return 0.0

        peaks_valleys = 0
        for i in range(1, len(data) - 1):
            if (
                data["close"].iloc[i] > data["close"].iloc[i - 1]
                and data["close"].iloc[i] > data["close"].iloc[i + 1]
            ) or (
                data["close"].iloc[i] < data["close"].iloc[i - 1]
                and data["close"].iloc[i] < data["close"].iloc[i + 1]
            ):
                peaks_valleys += 1

        return peaks_valleys / len(data) if len(data) > 0 else 0.0

    def _test_macd_effectiveness(self, data: pd.DataFrame) -> float:
        """測試MACD指標有效性"""
        if len(data) < 50:
            return 0.5

        macd_data = calculate_macd(data)
        macd = macd_data["macd"]
        signal = macd_data["macd_signal"]

        successful_signals = 0
        total_signals = 0

        for i in range(1, len(macd) - 5):
            if (
                macd.iloc[i] > signal.iloc[i] and macd.iloc[i - 1] <= signal.iloc[i - 1]
            ):  # 金叉
                future_returns = data["close"].iloc[i + 1 : i + 6].pct_change().sum()
                if future_returns > 0:
                    successful_signals += 1
                total_signals += 1
            elif (
                macd.iloc[i] < signal.iloc[i] and macd.iloc[i - 1] >= signal.iloc[i - 1]
            ):  # 死叉
                future_returns = data["close"].iloc[i + 1 : i + 6].pct_change().sum()
                if future_returns < 0:
                    successful_signals += 1
                total_signals += 1

        return successful_signals / total_signals if total_signals > 0 else 0.5

    def _calculate_gap_frequency(self, data: pd.DataFrame) -> float:
        """計算跳空頻率"""
        if len(data) < 2:
            return 0.0

        gaps = 0
        for i in range(1, len(data)):
            gap_up = data["low"].iloc[i] > data["high"].iloc[i - 1]
            gap_down = data["high"].iloc[i] < data["low"].iloc[i - 1]
            if gap_up or gap_down:
                gaps += 1

        return gaps / len(data) if len(data) > 0 else 0.0

    def _analyze_support_resistance(self, data: pd.DataFrame) -> float:
        """分析支撐阻力強度"""
        if len(data) < 20:
            return 0.5

        try:
            # 簡化的支撐阻力分析
            low_min = data["low"].min()
            high_max = data["high"].max()

            if pd.isna(low_min) or pd.isna(high_max) or low_min >= high_max:
                return 0.5

            price_levels = np.linspace(low_min, high_max, 20)
            level_touches = []

            for level in price_levels:
                touches = 0
                if level != 0:  # 避免除零錯誤
                    for _, row in data.iterrows():
                        if (
                            abs(row["low"] - level) / level < 0.02
                            or abs(row["high"] - level) / level < 0.02
                        ):
                            touches += 1
                level_touches.append(touches)

            if not level_touches:
                return 0.5

            max_touches = max(level_touches)
            return min(1.0, max_touches / len(data)) if len(data) > 0 else 0.5

        except Exception as e:
            print(f"支撐阻力分析錯誤: {e}")
            return 0.5

    def get_strategy_description(self) -> str:
        """返回策略描述"""
        return f"""
        LLM智能策略 (LLM Smart Strategy) - 自適應參數優化版
        
        🧠 智能特性：
        • 自動分析股票特性（3-6個月歷史數據）
        • 根據波動性、趨勢性、反轉頻率等特徵動態優化技術指標參數
        • 無需手動調參，策略自動適應不同股性
        
        📊 股性分析維度：
        • 波動性分析：年化波動率、波動性的波動性
        • 趨勢特性：趨勢一致性、趨勢強度、連續移動傾向
        • 反轉特性：反轉頻率、突破傾向
        • 技術指標響應性：MACD、均線、布林帶有效性測試
        • 綜合股性分類：高波動趨勢型、穩健趨勢型、震盪型等
        
        ⚙️ 當前參數設置（基於配置）：
        - 信心度閾值: {self.confidence_threshold}
        - 趨勢回望期: {self.trend_lookback}天
        - 事件觸發閾值: {self.event_threshold}
        - 每日最大交易數: {self.max_daily_trades}
        - 使用技術過濾: {"是" if self.use_technical_filter else "否"}
        
        🔧 技術指標（動態優化）：
        - MACD快線: {self.macd_fast} (根據趨勢性自動調整)
        - MACD慢線: {self.macd_slow} (根據趨勢性自動調整)
        - 短期均線: {self.ma_short}天 (根據反轉頻率自動調整)
        - 長期均線: {self.ma_long}天 (根據反轉頻率自動調整)
        
        🎯 工作流程：
        1. 深度分析股票特性，生成股性檔案
        2. 根據股性智能優化所有技術指標參數
        3. 事件驅動檢測關鍵技術信號
        4. LLM綜合分析做出交易決策
        5. 嚴格風險控制和信心度過濾
        
        ✨ 適用場景：
        - 全市場股票自動適應
        - 無需人工調參的智能交易
        - 適合不同股性的個股
        - 中短期交易策略
        """

    @classmethod
    def get_default_config(cls) -> StrategyConfig:
        """返回預設配置"""
        return StrategyConfig(
            name="LLM智能策略",
            description="基於股性分析的自適應LLM交易策略，自動優化技術指標參數",
            parameters={
                "confidence_threshold": 0.6,  # 降低閾值讓更多信號通過
                "trend_lookback": 20,
                "event_threshold": 0.05,
                "max_daily_trades": 3,
                "use_technical_filter": True,
                "ma_short": 10,  # 基準值，實際使用時會根據股性調整
                "ma_long": 20,  # 基準值，實際使用時會根據股性調整
            },
            parameter_specs={
                "confidence_threshold": ParameterSpec(
                    name="confidence_threshold",
                    display_name="LLM信心度閾值",
                    description="LLM決策的最低信心度要求",
                    param_type=ParameterType.FLOAT,
                    default_value=0.6,  # 降低預設值
                    min_value=0.3,
                    max_value=0.95,
                    step=0.05,
                ),
                "trend_lookback": ParameterSpec(
                    name="trend_lookback",
                    display_name="趨勢回望期",
                    description="趨勢分析的回望天數",
                    param_type=ParameterType.INTEGER,
                    default_value=20,
                    min_value=10,
                    max_value=50,
                    step=1,
                ),
                "event_threshold": ParameterSpec(
                    name="event_threshold",
                    display_name="事件觸發閾值",
                    description="關鍵事件的觸發敏感度",
                    param_type=ParameterType.FLOAT,
                    default_value=0.05,
                    min_value=0.01,
                    max_value=0.2,
                    step=0.01,
                ),
                "max_daily_trades": ParameterSpec(
                    name="max_daily_trades",
                    display_name="每日最大交易數",
                    description="每日允許的最大交易次數",
                    param_type=ParameterType.INTEGER,
                    default_value=3,
                    min_value=1,
                    max_value=10,
                    step=1,
                ),
                "use_technical_filter": ParameterSpec(
                    name="use_technical_filter",
                    display_name="技術指標過濾",
                    description="是否使用技術指標過濾信號",
                    param_type=ParameterType.BOOLEAN,
                    default_value=True,
                ),
                "ma_short": ParameterSpec(
                    name="ma_short",
                    display_name="短期均線基準",
                    description="短期移動平均線基準週期（實際使用時會根據股票反轉頻率自動調整）",
                    param_type=ParameterType.INTEGER,
                    default_value=10,
                    min_value=5,
                    max_value=20,
                    step=1,
                ),
                "ma_long": ParameterSpec(
                    name="ma_long",
                    display_name="長期均線基準",
                    description="長期移動平均線基準週期（實際使用時會根據股票反轉頻率自動調整）",
                    param_type=ParameterType.INTEGER,
                    default_value=20,
                    min_value=15,
                    max_value=50,
                    step=1,
                ),
            },
            risk_level="medium",
            market_type="all",
            strategy_type="ai_adaptive",
            category="intelligent",
        )

    def _log_daily_analysis(
        self,
        timestamp: pd.Timestamp,
        historical_data: pd.DataFrame,
        i: int,
        events: List[Dict[str, Any]],
        relevant_events: List[Dict[str, Any]],
        trend_analysis: Any,
        llm_decision: Dict[str, Any] = None,
        comprehensive_context: Dict[str, Any] = None,  # 新增參數
    ):
        """
        記錄每日分析數據到日誌

        Args:
            timestamp: 當前時間戳
            historical_data: 歷史數據
            i: 當前數據索引
            events: 所有檢測到的事件
            relevant_events: 相關事件
            trend_analysis: 趨勢分析結果
            llm_decision: LLM決策結果
            comprehensive_context: 全面技術分析上下文
        """
        try:
            current_row = historical_data.iloc[i]
            current_date = timestamp.strftime("%Y-%m-%d")

            # 準備市場數據
            market_data = {
                "price": float(current_row.get("close", current_row.get("Close", 0))),
                "volume": int(current_row.get("volume", current_row.get("Volume", 0))),
                "high": float(current_row.get("high", current_row.get("High", 0))),
                "low": float(current_row.get("low", current_row.get("Low", 0))),
                "open": float(current_row.get("open", current_row.get("Open", 0))),
            }

            # 計算日收益率
            if i > 0:
                prev_close = historical_data.iloc[i - 1].get(
                    "close",
                    historical_data.iloc[i - 1].get("Close", market_data["price"]),
                )
                market_data["daily_return"] = (
                    market_data["price"] - prev_close
                ) / prev_close
            else:
                market_data["daily_return"] = 0.0

            # 計算波動率（使用過去10天的標準差）
            if i >= 10:
                recent_returns = []
                for j in range(max(0, i - 9), i + 1):
                    if j > 0:
                        curr_price = historical_data.iloc[j].get(
                            "close", historical_data.iloc[j].get("Close", 0)
                        )
                        prev_price = historical_data.iloc[j - 1].get(
                            "close",
                            historical_data.iloc[j - 1].get("Close", curr_price),
                        )
                        if prev_price > 0:
                            daily_ret = (curr_price - prev_price) / prev_price
                            recent_returns.append(daily_ret)

                if recent_returns:
                    import numpy as np

                    market_data["volatility"] = float(np.std(recent_returns))
                else:
                    market_data["volatility"] = 0.0
            else:
                market_data["volatility"] = 0.0

            # 準備趨勢分析數據
            trend_data = None
            if trend_analysis:
                trend_data = {
                    "short_term": getattr(
                        trend_analysis, "short_term_trend", "neutral"
                    ),
                    "medium_term": getattr(
                        trend_analysis, "medium_term_trend", "neutral"
                    ),
                    "long_term": getattr(trend_analysis, "dominant_trend", "neutral"),
                    "trend_strength": getattr(trend_analysis, "trend_strength", 0.5),
                    "confidence": getattr(trend_analysis, "confidence", 0.5),
                }

                # 添加支撑阻力位信息
                if hasattr(trend_analysis, "support_resistance"):
                    sr = trend_analysis.support_resistance
                    trend_data["support_level"] = getattr(sr, "support", None)
                    trend_data["resistance_level"] = getattr(sr, "resistance", None)

            # 準備事件數據
            triggered_events_data = []
            for event in events:
                event_data = {
                    "event_type": event.get("type", "unknown"),
                    "severity": self._determine_event_severity(event),
                    "description": event.get(
                        "description", f"{event.get('type', 'unknown')} 事件"
                    ),
                    "technical_data": {
                        "indicator": event.get("indicator", event.get("type")),
                        "value": event.get("value"),
                        "threshold": event.get("threshold"),
                        "strength": event.get("strength", "medium"),
                    },
                }
                triggered_events_data.append(event_data)

            # 準備LLM決策數據
            llm_decision_data = None
            if llm_decision:
                llm_decision_data = {
                    "decision_made": True,
                    "prompt_version": self.strategy_type,
                    "decision_type": llm_decision.get("action", "HOLD"),
                    "confidence": llm_decision.get("confidence", 0.0),
                    "reasoning": llm_decision.get("reasoning", ""),
                    "key_factors": llm_decision.get("factors", []),
                    "raw_response": llm_decision.get("raw_response", ""),
                }
            else:
                llm_decision_data = {
                    "decision_made": False,
                    "reason": "No significant events or filtered out",
                }

            # 準備策略狀態數據
            strategy_state_data = {
                "position": "long" if self.current_position else "neutral",
                "cash": self.cash,
                "portfolio_value": self.current_portfolio_value,
                "shares": self.shares,
                "entry_price": self.position_entry_price
                if self.current_position
                else None,
                "trade_count_today": self.daily_trade_count,
                "total_trades": self.total_trades,
                "winning_trades": self.winning_trades,
            }

            # 計算當前損益
            if self.current_position and self.shares > 0:
                current_value = self.shares * market_data["price"]
                entry_value = self.shares * self.position_entry_price
                strategy_state_data["unrealized_pnl"] = current_value - entry_value
                strategy_state_data["unrealized_pnl_pct"] = (
                    current_value - entry_value
                ) / entry_value
            else:
                strategy_state_data["unrealized_pnl"] = 0.0
                strategy_state_data["unrealized_pnl_pct"] = 0.0

            # 記錄到日誌
            log_id = self.backtest_logger.log_daily_analysis(
                symbol=self.current_symbol or "UNKNOWN",
                date=current_date,
                market_data=market_data,
                trend_analysis=trend_data,
                comprehensive_technical_analysis=comprehensive_context,  # 新增參數
                triggered_events=triggered_events_data,
                llm_decision=llm_decision_data,
                trading_signal=None,  # 會在生成信號時單獨更新
                strategy_state=strategy_state_data,
            )

            # 記錄個別事件分析
            for event in events:
                if event.get("type"):  # 確保事件有類型
                    self.backtest_logger.log_event_analysis(
                        daily_log_id=log_id,
                        event_type=event.get("type"),
                        severity=self._determine_event_severity(event),
                        market_context={
                            "price_before": market_data["price"],
                            "volume": market_data["volume"],
                            "trend": trend_data.get("short_term", "neutral")
                            if trend_data
                            else "neutral",
                        },
                        llm_response={
                            "triggered_decision": llm_decision is not None,
                            "action_taken": llm_decision.get("action", "HOLD")
                            if llm_decision
                            else "NONE",
                            "confidence": llm_decision.get("confidence", 0.0)
                            if llm_decision
                            else 0.0,
                        },
                    )

            logger.debug(f"✅ 已記錄 {current_date} 的分析數據 (log_id: {log_id})")

        except Exception as e:
            logger.error(f"❌ 記錄日誌失敗: {e}")
            import traceback

            traceback.print_exc()

    def _determine_event_severity(self, event: Dict[str, Any]) -> str:
        """
        判斷事件嚴重程度

        Args:
            event: 事件字典

        Returns:
            嚴重程度: 'high', 'medium', 'low'
        """
        event_type = event.get("type", "").lower()
        strength = event.get("strength", "medium").lower()

        # 根據事件類型和強度判斷嚴重程度
        if strength == "high" or event_type in [
            "large_drop",
            "large_gain",
            "volume_spike",
        ]:
            return "high"
        elif strength == "low" or event_type in ["minor_support", "minor_resistance"]:
            return "low"
        else:
            return "medium"

    def _log_trading_signal(
        self,
        timestamp: pd.Timestamp,
        signal: "TradingSignal",
        llm_decision: Dict[str, Any],
    ):
        """
        記錄交易信號到日誌

        Args:
            timestamp: 信號時間戳
            signal: 交易信號對象
            llm_decision: LLM決策結果
        """
        try:
            current_date = timestamp.strftime("%Y-%m-%d")

            # 查找當天的日誌記錄
            recent_logs = self.backtest_logger.query_logs(
                symbol=self.current_symbol,
                date_from=current_date,
                date_to=current_date,
                limit=1,
            )

            if recent_logs:
                log_id = recent_logs[0]["id"]

                # 準備交易信號數據
                signal_data = {
                    "signal_type": signal.signal_type.name,
                    "price": signal.price,
                    "quantity": signal.quantity,
                    "confidence": signal.confidence,
                    "reasoning": signal.reason,
                    "stop_loss": signal.stop_loss,
                    "take_profit": signal.take_profit,
                    "timestamp": timestamp.isoformat(),
                    "llm_factors": llm_decision.get("factors", []),
                    "llm_confidence": llm_decision.get("confidence", 0.0),
                }

                # 更新當天的記錄
                with sqlite3.connect(self.backtest_logger.db_path) as conn:
                    conn.execute(
                        """
                        UPDATE daily_analysis_logs 
                        SET trading_signal = ?
                        WHERE id = ?
                    """,
                        (json.dumps(signal_data), log_id),
                    )

                logger.debug(f"✅ 已更新交易信號日誌 (log_id: {log_id})")

        except Exception as e:
            logger.error(f"❌ 記錄交易信號失敗: {e}")

    def get_backtest_summary(self) -> Dict[str, Any]:
        """
        獲取回測摘要

        Returns:
            回測摘要數據
        """
        if not self.backtest_logger:
            return {}

        return self.backtest_logger.get_session_summary()

    def export_backtest_logs(self, filepath: str = None):
        """
        導出回測日誌

        Args:
            filepath: 導出文件路徑，如果不提供則使用默認路徑
        """
        if not self.backtest_logger:
            logger.warning("日誌記錄器未啟用，無法導出")
            return

        if not filepath:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = f"backtest_logs_{self.current_symbol}_{timestamp}.json"

        self.backtest_logger.export_to_json(filepath)
        return filepath
