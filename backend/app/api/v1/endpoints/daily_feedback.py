"""
Daily Decision Improvement API
針對特定日期的決策改善建議
"""

import json
import os
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.llm.client import get_llm_client
from app.utils.backtest_logger import BacktestLogger
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

router = APIRouter()


class DailyFeedbackRequest(BaseModel):
    feedback: str
    date: str  # YYYY-MM-DD
    symbol: str = None  # Optional symbol, if not provided will search all symbols


class DailyImprovementResponse(BaseModel):
    analysis: str
    suggestions: List[str]


@router.post("/daily-feedback", response_model=DailyImprovementResponse)
async def analyze_daily_decision(
    request: DailyFeedbackRequest,
    db_path: str = Query(None, description="Database path"),
) -> DailyImprovementResponse:
    """
    分析特定日期的決策並提供改善建議
    Uses the same data access pattern as the working backtest_analysis API.
    """
    try:
        # 1. Use consistent path across the application
        if not db_path:
            db_path = "backend/data/backtest_logs.db"

        print(f"🔍 Analysis date: {request.date}")
        print(f"📝 User feedback: {request.feedback}")
        print(f"🗄️ Database path: {db_path}")

        if not Path(db_path).exists():
            print(f"❌ 數據庫文件不存在: {db_path}")
            raise HTTPException(
                status_code=404, detail=f"Database not found: {db_path}"
            )

            # 2. Find all sessions that have data for the target date
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # First, get all sessions that have data for this date
        cursor.execute(
            """
            SELECT DISTINCT session_id, symbol FROM daily_analysis_logs 
            WHERE date = ?
            ORDER BY session_id DESC
        """,
            (request.date,),
        )
        date_sessions = cursor.fetchall()

        if not date_sessions:
            conn.close()
            raise HTTPException(
                status_code=404, detail=f"未找到 {request.date} 的任何交易數據"
            )

        print(f"📊 該日期的session和股票: {date_sessions}")

        # Determine which session to use
        target_session = None
        target_symbol = None

        if request.symbol:
            # If user specified a symbol, find the session for that symbol
            for session_id, symbol in date_sessions:
                if request.symbol.upper() in symbol or symbol in request.symbol.upper():
                    target_session = session_id
                    target_symbol = symbol
                    break

            if not target_session:
                available_symbols = [symbol for _, symbol in date_sessions]
                conn.close()
                raise HTTPException(
                    status_code=404,
                    detail=f"未找到股票 {request.symbol} 在 {request.date} 的數據。可用股票: {', '.join(available_symbols)}",
                )
        else:
            # If no symbol specified, prioritize NVDA, TSLA, then others
            priority_symbols = ["NVDA", "TSLA", "AAPL", "MSFT"]

            for priority_symbol in priority_symbols:
                for session_id, symbol in date_sessions:
                    if priority_symbol in symbol:
                        target_session = session_id
                        target_symbol = symbol
                        break
                if target_session:
                    break

            if not target_session:
                # Fallback to first available
                target_session, target_symbol = date_sessions[0]

        print(f"✅ 使用session: {target_session}")
        print(f"🎯 查詢股票: {target_symbol}")
        conn.close()

        # 3. Initialize BacktestLogger and query data
        logger = BacktestLogger(db_path, session_id=target_session)
        logs = logger.query_logs(
            symbol=target_symbol, date_from=request.date, date_to=request.date, limit=1
        )

        if not logs:
            print(
                f"❌ 未找到指定日期的數據: {request.date} (session: {target_session})"
            )
            raise HTTPException(
                status_code=404, detail=f"未找到 {request.date} 的交易數據"
            )

        daily_data = logs[0]
        print(
            f"✅ 成功獲取交易數據: {target_symbol} - {len(daily_data.get('triggered_events', []))} 個技術事件"
        )

        # 5. 讀取交易策略內容
        strategy_content = load_trading_strategy()

        # 6. 使用LLM分析並生成改善建議
        improvement_response = await generate_daily_improvement_analysis(
            request.feedback, request.date, daily_data, strategy_content
        )

        return improvement_response

    except HTTPException:
        raise
    except Exception as e:
        import traceback

        error_msg = (
            f"處理反饋時發生錯誤: {str(e)}\n\nTraceback:\n{traceback.format_exc()}"
        )
        print(f"❌ {error_msg}")
        raise HTTPException(status_code=500, detail=f"分析失敗: {str(e)}")


# Remove the old get_daily_trading_data function - will be replaced with simpler logic


def load_trading_strategy() -> str:
    """
    讀取交易策略文件內容
    """
    try:
        # 構建策略文件路徑 - 修正路徑計算
        # 當前文件: backend/app/api/v1/endpoints/daily_feedback.py
        # 目標文件: backend/app/llm/strategies/prompt/traditional_strategy.md
        current_file = Path(__file__)  # daily_feedback.py
        app_dir = current_file.parent.parent.parent.parent  # 到達 backend/app/
        strategy_path = (
            app_dir / "llm" / "strategies" / "prompt" / "traditional_strategy.md"
        )

        print(f"📋 讀取策略文件: {strategy_path}")

        if not strategy_path.exists():
            print(f"⚠️ 策略文件不存在: {strategy_path}")
            print(f"🔍 檢查的路徑: {strategy_path.absolute()}")
            return "策略文件未找到"

        with open(strategy_path, "r", encoding="utf-8") as f:
            content = f.read()

        print(f"✅ 策略文件讀取成功，長度: {len(content)} 字符")
        return content

    except Exception as e:
        print(f"❌ 策略文件讀取錯誤: {e}")
        return f"策略文件讀取失敗: {str(e)}"


async def generate_daily_improvement_analysis(
    feedback: str, target_date: str, daily_data: Dict[str, Any], strategy_content: str
) -> DailyImprovementResponse:
    """
    生成日別決策改善分析
    """
    try:
        # 準備數據摘要
        triggered_events = daily_data.get("triggered_events", [])
        llm_decision = daily_data.get("llm_decision", {})
        symbol = daily_data.get("symbol", "Unknown")
        price = daily_data.get("price", 0)

        # 構建LLM提示
        context = f"""嗨！我是你的AI交易策略討論夥伴。用戶對 {target_date} 這天的決策有想法，讓我們一起分析並優化策略文件！

=== 用戶的想法 ===
{feedback}

=== 那天的情況 ===
{target_date} - {symbol} ${price:.2f}

=== 我當時的決策邏輯 ===
{llm_decision.get("decision_type", "N/A")}: {llm_decision.get("reasoning", "N/A")}

=== 當前交易策略文件內容 ===
{strategy_content}

請你詳細分析並提供具體可執行的策略文件修改建議！

## 我的看法
[先解釋當前策略為什麼會做出這個決策，再評估用戶建議的合理性，大約2-3段]

## 策略文件修改建議
請提供3個具體的修改建議，每個建議包含：
- 修改位置/章節
- 具體的新規則文字
- 實際的參數或條件

格式如下：
1. [修改標題]: [詳細說明要在策略文件的哪個部分添加/修改什麼具體規則，包括參數、條件、邏輯等完整內容，至少2-3行詳細描述]

2. [修改標題]: [詳細說明要在策略文件的哪個部分添加/修改什麼具體規則，包括參數、條件、邏輯等完整內容，至少2-3行詳細描述]

3. [修改標題]: [詳細說明要在策略文件的哪個部分添加/修改什麼具體規則，包括參數、條件、邏輯等完整內容，至少2-3行詳細描述]

## 修改原因說明
[簡要說明為什麼需要這些修改，以及預期的改善效果]
"""

        # 獲取LLM回應
        # Respect provider override if configured
        from app.config import settings as _settings
        _provider = (
            str(_settings.LLM_PROVIDER).strip().lower()
            if getattr(_settings, "LLM_PROVIDER", None)
            else None
        )
        if _provider not in {"azure", "openai", "gemini"}:
            _provider = None
        llm_client = get_llm_client(provider=_provider, temperature=0.7, max_tokens=1500)
        response = llm_client.invoke(context)
        response_text = (
            response.content if hasattr(response, "content") else str(response)
        )

        # 解析回應
        analysis_parts = parse_llm_response(response_text)

        return DailyImprovementResponse(
            analysis=analysis_parts.get("analysis", "分析生成中..."),
            suggestions=analysis_parts.get("suggestions", ["請稍後重試"]),
        )

    except Exception as e:
        print(f"❌ LLM分析錯誤: {e}")
        # 返回備用回應
        return DailyImprovementResponse(
            analysis=f"基於您的反饋「{feedback}」，我們正在分析 {target_date} 的決策合理性。",
            suggestions=[
                "檢查技術指標的組合使用",
                "評估市場趨勢的判斷準確性",
                "優化風險控制參數",
            ],
            strategy_review="策略需要根據市場變化持續優化調整。",
        )


def parse_llm_response(response_text: str) -> Dict[str, Any]:
    """
    解析LLM回應文本
    """
    try:
        print(f"🔍 原始LLM回應:\n{response_text}\n")

        parts = {"analysis": "", "suggestions": [], "strategy_review": ""}

        # 簡單的文本分割解析
        sections = response_text.split("##")
        print(f"📝 分割後sections數量: {len(sections)}")

        for i, section in enumerate(sections):
            section = section.strip()
            print(f"Section {i}: {section[:100]}...")

            if "我的看法" in section:
                # 移除標題並保留內容
                content = section.replace("我的看法", "", 1).strip()
                # 移除可能的冒號
                if content.startswith(":"):
                    content = content[1:].strip()
                parts["analysis"] = content
                print(f"✅ 找到「我的看法」: {content[:50]}...")
            elif "決策分析" in section:  # 保持向後相容
                content = section.replace("決策分析", "", 1).strip()
                if content.startswith(":"):
                    content = content[1:].strip()
                parts["analysis"] = content
            elif "策略文件修改建議" in section:
                suggestions_text = section.replace("策略文件修改建議", "", 1).strip()
                if suggestions_text.startswith(":"):
                    suggestions_text = suggestions_text[1:].strip()
                print(f"✅ 找到「策略文件修改建議」: {suggestions_text[:100]}...")

                # 更智能的建議提取 - 保留完整內容
                suggestions = []
                lines = suggestions_text.split("\n")
                current_suggestion = ""

                for line in lines:
                    line = line.strip()
                    # 檢查是否是新的建議項目開始
                    if line and any(line.startswith(f"{i}.") for i in range(1, 10)):
                        # 如果有之前的建議，先保存
                        if current_suggestion:
                            suggestions.append(current_suggestion.strip())
                        # 開始新的建議，去掉編號
                        current_suggestion = line[2:].strip()
                    elif line and current_suggestion:
                        # 繼續當前建議的內容
                        current_suggestion += "\n" + line
                    elif not current_suggestion and line:
                        # 處理沒有編號的建議行
                        current_suggestion = line

                # 添加最後一個建議
                if current_suggestion:
                    suggestions.append(current_suggestion.strip())

                print(f"📋 提取到完整建議數量: {len(suggestions)}")
                for i, suggestion in enumerate(suggestions):
                    print(f"建議 {i + 1}: {suggestion[:50]}...")

                parts["suggestions"] = suggestions
            elif "一些建議" in section:  # 保持向後相容
                suggestions_text = section.replace("一些建議", "", 1).strip()
                if suggestions_text.startswith(":"):
                    suggestions_text = suggestions_text[1:].strip()
                # 提取列表項目
                suggestions = []
                for line in suggestions_text.split("\n"):
                    line = line.strip()
                    if line and (
                        line.startswith("1.")
                        or line.startswith("2.")
                        or line.startswith("3.")
                        or line.startswith("4.")
                        or line.startswith("5.")
                    ):
                        suggestions.append(line[2:].strip())
                parts["suggestions"] = suggestions
            elif "改善建議" in section:  # 保持向後相容
                suggestions_text = section.replace("改善建議", "", 1).strip()
                if suggestions_text.startswith(":"):
                    suggestions_text = suggestions_text[1:].strip()
                suggestions = []
                for line in suggestions_text.split("\n"):
                    line = line.strip()
                    if line and (
                        line.startswith("1.")
                        or line.startswith("2.")
                        or line.startswith("3.")
                        or line.startswith("4.")
                        or line.startswith("5.")
                    ):
                        suggestions.append(line[2:].strip())
                parts["suggestions"] = suggestions
            elif "修改原因說明" in section:
                content = section.replace("修改原因說明", "", 1).strip()
                if content.startswith(":"):
                    content = content[1:].strip()
                parts["strategy_review"] = content
            elif "策略優化想法" in section:  # 保持向後相容
                content = section.replace("策略優化想法", "", 1).strip()
                if content.startswith(":"):
                    content = content[1:].strip()
                parts["strategy_review"] = content
            elif "策略檢討" in section:  # 保持向後相容
                content = section.replace("策略檢討", "", 1).strip()
                if content.startswith(":"):
                    content = content[1:].strip()
                parts["strategy_review"] = content

        print(
            f"📊 最終解析結果: analysis={bool(parts['analysis'])}, suggestions={len(parts['suggestions'])}, strategy_review={bool(parts['strategy_review'])}"
        )
        return parts

    except Exception as e:
        print(f"❌ 回應解析錯誤: {e}")
        return {
            "analysis": response_text[:200] + "...",
            "suggestions": ["請稍後重試"],
            "strategy_review": "檢討分析中...",
        }
