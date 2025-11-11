"""
Backtest analysis endpoints for retrieving and analyzing historical backtest data.
"""

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.llm.client import get_llm_client
from app.utils.backtest_logger import BacktestLogger
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

router = APIRouter()


# Response models
class MarketData(BaseModel):
    open: Optional[float] = None
    high: Optional[float] = None
    low: Optional[float] = None
    close: float
    volume: Optional[int] = None


class TechnicalEvent(BaseModel):
    event_type: str
    severity: str
    description: str
    technical_data: Optional[Dict[str, Any]] = None


class TrendAnalysis(BaseModel):
    short_term: Optional[str] = None
    medium_term: Optional[str] = None
    long_term: Optional[str] = None
    trend_strength: Optional[float] = None
    confidence: Optional[float] = None


class ComprehensiveTechnicalAnalysis(BaseModel):
    """Comprehensive technical analysis data structure"""

    date: Optional[str] = None
    price_action: Optional[Dict[str, Any]] = None
    moving_averages: Optional[Dict[str, Any]] = None
    volume_analysis: Optional[Dict[str, Any]] = None
    volatility_analysis: Optional[Dict[str, Any]] = None
    momentum_indicators: Optional[Dict[str, Any]] = None
    support_resistance: Optional[Dict[str, Any]] = None
    trend_analysis: Optional[Dict[str, Any]] = None
    market_regime: Optional[Dict[str, Any]] = None
    bollinger_analysis: Optional[Dict[str, Any]] = None
    macd_analysis: Optional[Dict[str, Any]] = None


class LLMDecision(BaseModel):
    decision_made: bool
    decision_type: Optional[str] = None
    confidence: Optional[float] = None
    reasoning: Optional[str] = None
    risk_level: Optional[str] = None


class DayAnalysisData(BaseModel):
    date: str
    symbol: str
    price: float
    daily_return: Optional[float] = None
    volume: Optional[int] = None
    market_data: Optional[MarketData] = None
    trend_analysis: Optional[TrendAnalysis] = None
    comprehensive_technical_analysis: Optional[ComprehensiveTechnicalAnalysis] = (
        None  # Added field
    )
    technical_events: List[TechnicalEvent] = []
    llm_decision: Optional[LLMDecision] = None
    strategy_state: Optional[Dict[str, Any]] = None


class RetrospectiveAnalysis(BaseModel):
    llm_commentary: str
    decision_quality_score: Optional[float] = None
    alternative_perspective: Optional[str] = None
    lessons_learned: Optional[str] = None


class DayAnalysisResponse(BaseModel):
    historical_data: DayAnalysisData
    retrospective_analysis: Optional[RetrospectiveAnalysis] = None


class AvailableDatesResponse(BaseModel):
    dates: List[str]
    date_range: Dict[str, str]
    total_days: int


@router.get("/available-dates/{run_id}", response_model=AvailableDatesResponse)
async def get_available_dates_by_run_id(
    run_id: str,
    symbol: Optional[str] = Query(None, description="Filter by symbol"),
    db_path: str = Query("backend/data/backtest_logs.db", description="Database path"),
) -> AvailableDatesResponse:
    """
    Get available dates for a specific run_id by mapping it to session_id.
    """
    try:
        # Check if database exists
        if not Path(db_path).exists():
            raise HTTPException(status_code=404, detail="Database file not found")

        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Extract timestamp from run_id if it exists
        run_timestamp = None
        if "-" in run_id:
            parts = run_id.split("-")
            if len(parts) >= 2:
                try:
                    run_timestamp = int(parts[1])
                except ValueError:
                    pass

        # Try to find matching session_id by timestamp proximity
        actual_session_id = None
        if run_timestamp:
            # Convert milliseconds to seconds for matching
            run_timestamp_sec = run_timestamp // 1000
            # Try exact match first, then nearby timestamps
            for delta in [0, 1, -1, 2, -2]:
                search_timestamp = run_timestamp_sec + delta
                cursor.execute(
                    """
                    SELECT DISTINCT session_id FROM daily_analysis_logs 
                    WHERE session_id LIKE ?
                    ORDER BY session_id DESC
                """,
                    (f"%{search_timestamp}%",),
                )
                matching_sessions = cursor.fetchall()
                if matching_sessions:
                    actual_session_id = matching_sessions[0][0]
                    break

        if not actual_session_id:
            # If no matching session found, try to get the latest session
            cursor.execute(
                "SELECT DISTINCT session_id FROM daily_analysis_logs ORDER BY session_id DESC LIMIT 1"
            )
            latest_session = cursor.fetchone()
            if latest_session:
                actual_session_id = latest_session[0]
            else:
                conn.close()
                raise HTTPException(
                    status_code=404, detail=f"No sessions found in database"
                )

        # Get dates for the matched session
        query = "SELECT DISTINCT date FROM daily_analysis_logs WHERE session_id = ?"
        params = [actual_session_id]

        if symbol:
            query += " AND symbol = ?"
            params.append(symbol)

        query += " ORDER BY date DESC"

        cursor.execute(query, params)
        dates = [row[0] for row in cursor.fetchall()]

        # Get date range
        date_range = {}
        if dates:
            date_range = {
                "start": dates[-1],  # earliest date
                "end": dates[0],  # latest date
            }

        conn.close()

        return AvailableDatesResponse(
            dates=dates, date_range=date_range, total_days=len(dates)
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving dates: {str(e)}")


@router.get("/available-dates", response_model=AvailableDatesResponse)
async def get_available_dates(
    symbol: Optional[str] = Query(None, description="Filter by symbol"),
    db_path: str = Query("backend/data/backtest_logs.db", description="Database path"),
) -> AvailableDatesResponse:
    """
    Get all available dates from backtest logs.
    """
    try:
        # Check if database exists
        if not Path(db_path).exists():
            raise HTTPException(status_code=404, detail="Database file not found")

        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        query = "SELECT DISTINCT date FROM daily_analysis_logs"
        params = []

        if symbol:
            query += " WHERE symbol = ?"
            params.append(symbol)

        query += " ORDER BY date DESC"

        cursor.execute(query, params)
        dates = [row[0] for row in cursor.fetchall()]

        # Get date range
        date_range = {}
        if dates:
            date_range = {
                "start": dates[-1],  # earliest date
                "end": dates[0],  # latest date
            }

        conn.close()

        return AvailableDatesResponse(
            dates=dates, date_range=date_range, total_days=len(dates)
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving dates: {str(e)}")


@router.get("/analysis/day/{run_id}", response_model=DayAnalysisResponse)
async def get_day_analysis(
    run_id: str,
    date: str = Query(..., description="Date in YYYY-MM-DD format"),
    symbol: Optional[str] = Query(None, description="Filter by symbol"),
    db_path: str = Query("backend/data/backtest_logs.db", description="Database path"),
    include_retrospective: bool = Query(
        True, description="Include LLM retrospective analysis"
    ),
) -> DayAnalysisResponse:
    """
    Get detailed analysis for a specific date.
    """
    try:
        # Validate date format
        try:
            datetime.strptime(date, "%Y-%m-%d")
        except ValueError:
            raise HTTPException(
                status_code=400, detail="Invalid date format. Use YYYY-MM-DD"
            )

        # Check if database exists
        if not Path(db_path).exists():
            raise HTTPException(status_code=404, detail="Database file not found")

        # Debug: First check what session_ids exist in database
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Extract timestamp from run_id if it exists
        run_timestamp = None
        if "-" in run_id:
            parts = run_id.split("-")
            if len(parts) >= 2:
                try:
                    run_timestamp = int(parts[1])
                except ValueError:
                    pass

        # Try to find matching session_id by timestamp proximity (within 10 seconds)
        if run_timestamp:
            # Convert milliseconds to seconds for matching
            run_timestamp_sec = run_timestamp // 1000
            # Try exact match first, then nearby timestamps
            for delta in [0, 1, -1, 2, -2]:
                search_timestamp = run_timestamp_sec + delta
                cursor.execute(
                    """
                    SELECT DISTINCT session_id FROM daily_analysis_logs 
                    WHERE session_id LIKE ?
                    ORDER BY session_id DESC
                """,
                    (f"%{search_timestamp}%",),
                )
                matching_sessions = cursor.fetchall()
                if matching_sessions:
                    break
            else:
                matching_sessions = []
        else:
            cursor.execute("""
                SELECT DISTINCT session_id FROM daily_analysis_logs 
                ORDER BY session_id DESC
                LIMIT 1
            """)
            matching_sessions = cursor.fetchall()
        conn.close()

        if not matching_sessions:
            # If no matching session found, try to get the latest session
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute(
                "SELECT DISTINCT session_id FROM daily_analysis_logs ORDER BY session_id DESC LIMIT 1"
            )
            latest_session = cursor.fetchone()
            conn.close()

            if latest_session:
                actual_session_id = latest_session[0]
            else:
                raise HTTPException(
                    status_code=404, detail=f"No sessions found in database"
                )
        else:
            actual_session_id = matching_sessions[0][0]

        # Initialize logger with the actual session_id
        logger = BacktestLogger(db_path, session_id=actual_session_id)
        logs = logger.query_logs(symbol=symbol, date_from=date, date_to=date, limit=1)

        if not logs:
            raise HTTPException(
                status_code=404, detail=f"No data found for date {date}"
            )

        log = logs[0]

        # Parse market data
        market_data = None
        if log.get("market_data"):
            market_data = MarketData(**log["market_data"])
        else:
            market_data = MarketData(close=log["price"])

        # Parse trend analysis
        trend_analysis = None
        if log.get("trend_analysis"):
            trend_analysis = TrendAnalysis(**log["trend_analysis"])

        # Parse comprehensive technical analysis
        comprehensive_technical_analysis = None
        if log.get("comprehensive_technical_analysis"):
            comprehensive_technical_analysis = ComprehensiveTechnicalAnalysis(
                **log["comprehensive_technical_analysis"]
            )

        # Parse technical events
        technical_events = []
        if log.get("triggered_events"):
            for event in log["triggered_events"]:
                technical_events.append(TechnicalEvent(**event))

        # Parse LLM decision
        llm_decision = None
        if log.get("llm_decision"):
            llm_decision = LLMDecision(**log["llm_decision"])

        # Create historical data response
        historical_data = DayAnalysisData(
            date=log["date"],
            symbol=log["symbol"],
            price=log["price"],
            daily_return=log.get("daily_return"),
            volume=log.get("volume"),
            market_data=market_data,
            trend_analysis=trend_analysis,
            comprehensive_technical_analysis=comprehensive_technical_analysis,  # Added field
            technical_events=technical_events,
            llm_decision=llm_decision,
            strategy_state=log.get("strategy_state"),
        )

        # Generate retrospective analysis if requested
        retrospective_analysis = None
        if include_retrospective and llm_decision:
            retrospective_analysis = await generate_retrospective_analysis(
                historical_data
            )

        return DayAnalysisResponse(
            historical_data=historical_data,
            retrospective_analysis=retrospective_analysis,
        )

    except HTTPException:
        raise
    except Exception as e:
        import traceback

        error_detail = f"Error retrieving day analysis: {str(e)}\n\nTraceback:\n{traceback.format_exc()}"
        print(f"❌ API Error: {error_detail}")  # Add console logging
        raise HTTPException(status_code=500, detail=error_detail)


async def generate_retrospective_analysis(
    day_data: DayAnalysisData,
) -> RetrospectiveAnalysis:
    """
    Generate retrospective analysis using LLM with deeper market insight.
    Optimized for performance and reliability.
    """
    import asyncio

    try:
        print(f"🔍 Starting retrospective analysis: {day_data.date} {day_data.symbol}")

        # Set timeout for the entire analysis process
        timeout_seconds = 25  # Prevent hanging requests

        async def _generate_analysis():
            # Prepare context for LLM
            daily_return_text = (
                f"{day_data.daily_return:+.2%}" if day_data.daily_return else "N/A"
            )
            print(f"📊 Daily return: {daily_return_text}")

            # Format technical events with more detail
            if day_data.technical_events:
                events_text = "\n".join(
                    [
                        f"• {event.event_type} (Risk level: {event.severity}): {event.description}"
                        for event in day_data.technical_events[
                            :3
                        ]  # Limit to 3 events to reduce prompt size
                    ]
                )
                print(
                    f"📈 Number of technical events: {len(day_data.technical_events)}"
                )
            else:
                events_text = "No major technical events triggered today"
                print("📈 No technical events")

            # Format confidence and reasoning (truncated for performance)
            confidence_text = (
                f"{day_data.llm_decision.confidence:.1%}"
                if (day_data.llm_decision and day_data.llm_decision.confidence)
                else "No confidence data"
            )
            reasoning_text = (
                day_data.llm_decision.reasoning
                if (day_data.llm_decision and day_data.llm_decision.reasoning)
                else "No reasoning record"
            )
            decision_text = (
                day_data.llm_decision.decision_type
                if day_data.llm_decision
                else "No decision record"
            )
            risk_level = (
                day_data.llm_decision.risk_level
                if (day_data.llm_decision and day_data.llm_decision.risk_level)
                else "Not assessed"
            )
            # Removed verbose LLM progress logging for cleaner output

            # Simplified context for better performance
            try:
                context = f"""You are a professional trader reviewing this trading decision. Keep it concise with focused analysis.

=== Trading Context ===
Date: {day_data.date} | Symbol: {day_data.symbol} | Price: ${day_data.price:.2f} | Return: {daily_return_text}

=== Technical Events ===
{events_text}

=== Trend Analysis ===
Short-term: {day_data.trend_analysis.short_term if day_data.trend_analysis else "Not analyzed"} | Medium-term: {day_data.trend_analysis.medium_term if day_data.trend_analysis else "Not analyzed"} | Long-term: {day_data.trend_analysis.long_term if day_data.trend_analysis else "Not analyzed"}

=== AI Decision ===
Decision: {decision_text} | Confidence: {confidence_text} | Risk: {risk_level}
Reasoning: {reasoning_text[:200] if reasoning_text else "No reasoning record"}...

=== Please Analyze Concisely ===
1. **Environment Assessment**: What was the actual market situation? Major risks and opportunities?
2. **Decision Evaluation**: Was the AI decision reasonable? Any important omissions?  
3. **Practical Advice**: How would you operate? Specific entry, stop-loss, position recommendations?
4. **Experience Summary**: What is the core trading lesson from this case?

Please use professional but concise language, keeping within 800 words. Highlight insights and avoid lengthy descriptions.
"""
            except Exception as e:
                print(f"❌ LLM context building failed: {e}")
                context = f"Trading review analysis: {day_data.date} {day_data.symbol} - Data processing error"

            try:
                # Initialize LLM client with provider override if configured
                from app.config import settings as _settings
                _provider = (
                    str(_settings.LLM_PROVIDER).strip().lower()
                    if getattr(_settings, "LLM_PROVIDER", None)
                    else None
                )
                if _provider not in {"azure", "openai", "gemini"}:
                    _provider = None
                llm_client = get_llm_client(
                    provider=_provider, temperature=0.8, max_tokens=1000
                )  # Reduced token limit

                # Get LLM response with timeout handling
                response = llm_client.invoke(context)
                commentary = (
                    response.content if hasattr(response, "content") else str(response)
                )
                commentary = commentary.strip()

                # Limit commentary length to prevent oversized responses
                if len(commentary) > 3000:
                    commentary = (
                        commentary[:3000]
                        + "\n\n[Response truncated to ensure stable transmission]"
                    )

                return commentary
            except Exception as e:
                print(f"❌ LLM invocation failed: {e}")
                return f"LLM analysis generation failed: {str(e)}. Please try again later or contact technical support."

        # Execute with timeout
        try:
            commentary = await asyncio.wait_for(
                _generate_analysis(), timeout=timeout_seconds
            )
        except asyncio.TimeoutError:
            print(f"❌ Analysis timeout ({timeout_seconds} seconds)")
            commentary = "Analysis processing timeout, please try again later. This may be due to network latency or high system load."

        try:
            print(f"📋 Building retrospective analysis results...")
            result = RetrospectiveAnalysis(
                llm_commentary=commentary,
                decision_quality_score=None,  # No scoring needed
                alternative_perspective="Deep market analysis mode",
                lessons_learned="Trading insights based on practical experience",
            )
            print(f"✅ Retrospective analysis result built successfully")
            return result
        except Exception as e:
            print(f"❌ Retrospective analysis result build failed: {e}")
            return RetrospectiveAnalysis(
                llm_commentary=f"Analysis results construction failed: {str(e)}",
                decision_quality_score=None,
                alternative_perspective="Error Handling Mode",
                lessons_learned="System error needs to be fixed",
            )

    except Exception as e:
        print(f"❌ Overall analysis process failed: {e}")
        import traceback

        print(f"Detailed error trace: {traceback.format_exc()}")
        return RetrospectiveAnalysis(
            llm_commentary=f"Analysis system temporarily unavailable: {str(e)}",
            decision_quality_score=None,
            alternative_perspective="System Error",
            lessons_learned="Technical support needed",
        )


@router.get("/session-dates/{session_id}", response_model=AvailableDatesResponse)
async def get_session_dates(
    session_id: str,
    db_path: str = Query("backend/data/backtest_logs.db", description="Database path"),
) -> AvailableDatesResponse:
    """
    Get available dates for a specific session.
    """
    try:
        if not Path(db_path).exists():
            raise HTTPException(status_code=404, detail="Database file not found")

        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        cursor.execute(
            "SELECT DISTINCT date FROM daily_analysis_logs WHERE session_id = ? ORDER BY date DESC",
            (session_id,),
        )
        dates = [row[0] for row in cursor.fetchall()]

        date_range = {}
        if dates:
            date_range = {"start": dates[-1], "end": dates[0]}

        conn.close()

        return AvailableDatesResponse(
            dates=dates, date_range=date_range, total_days=len(dates)
        )

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error retrieving session dates: {str(e)}"
        )
