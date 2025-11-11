'use client'

import React, { useState, useRef, useCallback, useEffect } from 'react'
import dynamic from 'next/dynamic'
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Progress } from '@/components/ui/progress'
import { Alert, AlertDescription } from '@/components/ui/alert'
import { Badge } from '@/components/ui/badge'
// Removed lucide-react icons due to React 19 compatibility issues
// Using Unicode symbols instead
const BacktestResultsWithAnalysis = dynamic(
  () => import('@/components/analysis/BacktestResultsWithAnalysis'),
  { ssr: false }
)

interface StreamMessage {
  type: 'start' | 'progress' | 'trading_progress' | 'result' | 'complete' | 'error'
  message?: string
  step?: string
  day?: number
  total_days?: number
  progress?: number
  event_type?: string
  data?: any
  // 可能在根層級的performance_metrics
  performance_metrics?: {
    total_return: number
    win_rate: number
    max_drawdown: number
    total_trades: number
    total_value: number
    cash: number
    position_value: number
  }
  // 可能在根層級的pnl_status
  pnl_status?: {
    unrealized_pnl?: number
    unrealized_pnl_pct?: number
    holding_days?: number
    shares?: number
    risk_level?: string
    cash_remaining?: number
    total_value?: number
  }
  extra_data?: {
    pnl_status?: {
      unrealized_pnl?: number
      unrealized_pnl_pct?: number
      holding_days?: number
      shares?: number
      risk_level?: string
      cash_remaining?: number
      total_value?: number
    }
    performance_metrics?: {
      total_return: number
      win_rate: number
      max_drawdown: number
      total_trades: number
      total_value: number
      cash: number
      position_value: number
    }
  }
}

interface DynamicPerformance {
  total_return: number     // 總回報率（基於總價值）
  win_rate: number         // 勝率 (0-1)
  max_drawdown: number     // 最大回撤 (0-1)
  total_trades: number     // 完成的交易次數（有意義）
  total_realized_pnl?: number      // 累積實現損益
  cumulative_trade_return_rate?: number  // 累積交易收益率
  // 未來可添加：
  // avg_trade_return?: number    // 平均每筆交易收益率
  // profit_loss_ratio?: number   // 盈虧比
  // max_single_loss?: number     // 最大單筆虧損
}

interface PnLStatus {
  unrealized_pnl?: number
  unrealized_pnl_pct?: number
  holding_days?: number
  shares?: number
  risk_level?: string
  cash_remaining?: number
  total_value?: number
}

interface BacktestResult {
  trades: any[]
  performance: any
  stock_data: any[]
  signals: any[]
  llm_decisions: any[]
  statistics: {
    total_trades: number
    win_rate: number
    total_return: number
    max_drawdown: number
    final_value?: number
    total_realized_pnl?: number
    cumulative_trade_return_rate?: number
  }
}

export default function StreamingLLMRunner() {
  const [symbol, setSymbol] = useState('AAPL')
  const [period, setPeriod] = useState('1y')
  const [initialCapital] = useState(100000000) // 設置1億作為無上限資金，不影響純交易損益計算
  
  const [isRunning, setIsRunning] = useState(false)
  const [progress, setProgress] = useState(0)
  const [currentStep, setCurrentStep] = useState('')
  const [messages, setMessages] = useState<string[]>([])
  const [result, setResult] = useState<BacktestResult | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [isStarting, setIsStarting] = useState(false) // 新增：防止重複點擊
  const [currentRunId, setCurrentRunId] = useState<string | null>(null) // 新增：追踪當前回測的唯一標識
  
  // 動態績效狀態
  const [dynamicPerformance, setDynamicPerformance] = useState<DynamicPerformance>({
    total_return: 0,
    win_rate: 0,
    max_drawdown: 0,
    total_trades: 0
  })
  
  // P&L狀態
  const [pnlStatus, setPnlStatus] = useState<PnLStatus | null>(null)
  
  // 實時信號收集
  const [realTimeSignals, setRealTimeSignals] = useState<any[]>([])
  const [realTimeLLMDecisions, setRealTimeLLMDecisions] = useState<any[]>([])
  const [realTimeStockData, setRealTimeStockData] = useState<any[]>([])
  
  const eventSourceRef = useRef<EventSource | null>(null)
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const sessionIdRef = useRef<string | null>(null) // 添加會話 ID

  // 清理函數 - 確保 EventSource 正確關閉
  const cleanupEventSource = useCallback(() => {
    if (eventSourceRef.current) {
      console.log('清理 EventSource 連接')
      eventSourceRef.current.close()
      eventSourceRef.current = null
    }
    sessionIdRef.current = null
  }, [])

  // 組件卸載時清理
  useEffect(() => {
    return () => {
      cleanupEventSource()
    }
  }, [cleanupEventSource])

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }

  const addMessage = useCallback((message: string) => {
    setMessages(prev => [...prev, message])
    setTimeout(scrollToBottom, 100)
  }, [])

  const startStreaming = async () => {
    // 防止重複點擊
    if (isRunning || isStarting) {
      console.log('回測已在進行中，忽略重複請求')
      return
    }

    // 生成唯一的會話 ID 和 runId
    const sessionId = `session-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`
    const runId = `run-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`
    sessionIdRef.current = sessionId
    setCurrentRunId(runId)
    console.log('開始新的串流回測，會話 ID:', sessionId, 'Run ID:', runId)
    
    setIsStarting(true)
    
    // 先清理之前的連接
    cleanupEventSource()

    setIsRunning(true)
    setProgress(0)
    setCurrentStep('')
    setMessages([])
    setResult(null)
    setError(null)
    setPnlStatus(null) // 重置P&L狀態
    
    // 重置實時信號數據
    setRealTimeSignals([])
    setRealTimeLLMDecisions([])
    setRealTimeStockData([])

    const params = new URLSearchParams({
      symbol,
      period,
      session_id: sessionId, // 添加會話 ID
    })

    // 使用 Next.js 重寫規則代理到後端，避免跨域與開發覆蓋層問題
    const url = `/api/v1/llm-stream/llm-backtest-stream?${params}`
    
    try {
      console.log('創建新的 EventSource:', url)
      eventSourceRef.current = new EventSource(url)
      
      eventSourceRef.current.onopen = () => {
        console.log('EventSource 連接已建立')
        setIsStarting(false)
      }
      
      eventSourceRef.current.onmessage = (event) => {
        try {
          const data: StreamMessage = JSON.parse(event.data)
          console.log('收到串流數據:', data.type, data.event_type, data.message)
          
          // 調試：檢查performance_update事件
          if (data.event_type === 'performance_update') {
            console.log('Performance Update詳細數據:', {
              performance_metrics: data.performance_metrics,
              extra_data: data.extra_data,
              message: data.message
            })
          }
          
          switch (data.type) {
            case 'start':
              addMessage(data.message || '開始回測...')
              break
              
            case 'progress':
              setCurrentStep(data.message || '')
              // 只顯示重要的進度訊息，過濾內部處理訊息
              const progressMessage = data.message || ''
              if (!progressMessage.includes('正在獲取') && 
                  !progressMessage.includes('成功獲取') && 
                  !progressMessage.includes('初始化') &&
                  !progressMessage.includes('開始執行') &&
                  !progressMessage.includes('分析回測結果')) {
                addMessage(progressMessage)
              }
              break
              
              case 'trading_progress':
                if (data.total_days && data.day) {
                  const progressPercent = (data.day / data.total_days) * 100
                  setProgress(progressPercent)
                  
                  // 統一處理P&L數據更新 - 在所有事件類型前先更新
                  let pnlData = data.extra_data?.pnl_status || data.pnl_status
                  if (pnlData) {
                    console.log('更新P&L狀態:', {
                      event_type: data.event_type,
                      holding_days: pnlData.holding_days,
                      unrealized_pnl: pnlData.unrealized_pnl,
                      shares: pnlData.shares,
                      full_data: pnlData
                    })
                    setPnlStatus(pnlData as PnLStatus)
                  }
                  
                  if (data.event_type === 'llm_decision') {
                    // 保留完整的LLM決策內容，便於後續優化分析
                    const message = data.message || ''
                    addMessage(`🤖 ${message}`)
                    
                    // 收集LLM決策數據
                    if (data.extra_data) {
                      const llmDecision = {
                        date: new Date().toISOString(),
                        day: data.day,
                        decision: {
                          action: 'THINK', // LLM思考但不是買賣信號
                          confidence: 0.8,
                          reason: message
                        },
                        price: (data.extra_data as any).current_price || 0,
                        timestamp: new Date().toISOString()
                      }
                      setRealTimeLLMDecisions(prev => [...prev, llmDecision])
                    }
                  } else if (data.event_type === 'signal_generated') {
                    // 優化信號生成顯示
                    const message = data.message || ''
                    const signalMatch = message.match(/(BUY|SELL).*?信心度: ([\d.]+)/)
                    if (signalMatch) {
                      const signal = signalMatch[1]
                      const confidence = signalMatch[2]
                      const icon = signal === 'BUY' ? '🚀' : '📤'
                      addMessage(`${icon} 執行 ${signal} 信號 (信心度: ${confidence})`)
                      
                      // 收集交易信號數據
                      const tradingSignal = {
                        timestamp: new Date().toISOString(),
                        signal_type: signal,
                        price: (data.extra_data as any)?.current_price || 0,
                        confidence: parseFloat(confidence),
                        reason: message
                      }
                      setRealTimeSignals(prev => [...prev, tradingSignal])
                    } else {
                      addMessage(`📈 ${message}`)
                    }
                    
                    // 靜默更新績效數據，不重複顯示訊息（P&L數據已在上方統一更新）
                    let signalMetrics = data.extra_data?.performance_metrics || (data as any).performance_metrics
                    let strategyStats = (data as any).strategy_statistics || (data.extra_data as any)?.strategy_statistics
                    
                    if (signalMetrics) {
                      setDynamicPerformance({
                        total_return: signalMetrics.total_return || 0,
                        win_rate: strategyStats?.strategy_win_rate || signalMetrics.win_rate || 0,
                        max_drawdown: signalMetrics.max_drawdown || 0,
                        total_trades: strategyStats?.total_trades || signalMetrics.total_trades || 0,
                        total_realized_pnl: strategyStats?.total_realized_pnl || signalMetrics.total_realized_pnl || 0,
                        cumulative_trade_return_rate: strategyStats?.cumulative_trade_return_rate || signalMetrics.cumulative_trade_return_rate || 0
                      })
                    }
                  } else if (data.event_type === 'llm_skipped') {
                    // 跳過不重要的訊息，減少日誌雜訊
                    // addMessage(`⏭️ ${data.message}`)
                  } else if (data.event_type === 'entry_point') {
                    addMessage(`🚀 ${data.message}`)
                  } else if (data.event_type === 'exit_point') {
                    addMessage(`📤 ${data.message}`)
                  } else if (data.event_type === 'performance_update') {
                    // 優化績效更新邏輯，避免重複顯示
                    let metrics = data.extra_data?.performance_metrics || (data as any).performance_metrics
                    let strategyStats = (data as any).strategy_statistics || (data.extra_data as any)?.strategy_statistics
                    
                    if (metrics) {
                      const newTradeCount = strategyStats?.total_trades || metrics.total_trades || 0
                      const newReturn = metrics.total_return || 0
                      const newWinRate = strategyStats?.strategy_win_rate || metrics.win_rate || 0
                      
                      const prevTradeCount = dynamicPerformance.total_trades
                      const prevReturn = dynamicPerformance.total_return
                      
                      // 只在交易數量真正增加時顯示交易完成訊息
                      if (newTradeCount > prevTradeCount && newTradeCount > 0) {
                        const returnText = (newReturn * 100).toFixed(2)
                        const winRateText = (newWinRate * 100).toFixed(1)
                        addMessage(`💰 交易完成 | 總回報: ${returnText}% | 勝率: ${winRateText}% | 完成交易: ${newTradeCount}筆`)
                      } else if (newTradeCount === 0 && prevTradeCount === 0 && Math.abs(newReturn - prevReturn) > 0.05) {
                        // 只有在真正有收益率大幅變化且無交易時，才顯示績效更新（避免無意義的0%更新）
                        const returnText = (newReturn * 100).toFixed(2)
                        const winRateText = (newWinRate * 100).toFixed(1)
                        addMessage(`📊 績效更新 | 總回報: ${returnText}% | 勝率: ${winRateText}%`)
                      }
                      
                      setDynamicPerformance({
                        total_return: newReturn,
                        win_rate: newWinRate,
                        max_drawdown: metrics.max_drawdown || 0,
                        total_trades: newTradeCount,
                        total_realized_pnl: strategyStats?.total_realized_pnl || metrics.total_realized_pnl || 0,
                        cumulative_trade_return_rate: strategyStats?.cumulative_trade_return_rate || metrics.cumulative_trade_return_rate || 0
                      })
                    }
                    
                    // P&L狀態已在上方統一更新，此處不再重複更新
                  } else {
                    // 過濾系統訊息，只顯示重要內容
                    const message = data.message || ''
                    if (!message.includes('處理進度') && 
                        !message.includes('開始LLM分析') && 
                        message.trim() !== '') {
                      addMessage(message)
                    }
                  }
                }
                break
                
              case 'result':
              setResult(data.data)
              
              // 設置完整的股票數據用於圖表
              if (data.data.stock_data) {
                setRealTimeStockData(data.data.stock_data)
              }
              
              // 更新最終performance數據，優先使用statistics中的策略統計數據
              const finalStrategyStats = data.data.strategy_statistics || {}
              const finalPerformance = data.data.performance || {}
              const finalStatistics = data.data.statistics || {}
              
              setDynamicPerformance({
                total_return: finalStatistics.total_return || finalPerformance.total_return || 0,
                win_rate: finalStatistics.win_rate / 100 || finalStrategyStats.strategy_win_rate || finalPerformance.win_rate || 0, // 轉換百分比為小數
                max_drawdown: finalStatistics.max_drawdown || finalPerformance.max_drawdown || 0,
                total_trades: finalStatistics.total_trades || finalStrategyStats.total_trades || 0,
                total_realized_pnl: finalStatistics.total_realized_pnl || finalStrategyStats.total_realized_pnl || 0,
                cumulative_trade_return_rate: finalStatistics.total_return / 100 || finalStrategyStats.cumulative_trade_return_rate || 0 // 使用總回報率作為累積交易收益率
              })
              
              addMessage('✅ 回測完成，正在生成圖表...')
              break
              
            case 'complete':
              // 只顯示完成訊息，不顯示可能不準確的數據總結
              addMessage('🎉 回測完成！請查看下方圖卡獲取準確的統計數據')
              addMessage(data.message || '所有處理完成！')
              setIsRunning(false)
              cleanupEventSource()
              break
              
            case 'error':
              setError(data.message || '發生未知錯誤')
              addMessage(`❌ 錯誤: ${data.message}`)
              setIsRunning(false)
              cleanupEventSource()
              break
          }
        } catch (err) {
          console.error('解析串流數據錯誤:', err)
        }
      }
      
      eventSourceRef.current.onerror = (event) => {
        console.error('EventSource 錯誤:', event)
        setError('連接中斷或伺服器錯誤')
        setIsRunning(false)
        setIsStarting(false)
        cleanupEventSource()
      }
      
    } catch (err) {
      console.error('啟動串流錯誤:', err)
      setError('無法啟動串流回測')
      setIsRunning(false)
      setIsStarting(false)
    }
  }

  const stopStreaming = () => {
    console.log('手動停止串流')
    cleanupEventSource()
    setIsRunning(false)
    setIsStarting(false)
  }

  return (
    <div className="container mx-auto p-6 space-y-6">
      <div className="text-center">
        <h1 className="text-3xl font-bold gradient-text mb-2">
          🚀 串流式 LLM 策略回測
        </h1>
        <p className="text-gray-600">即時觀看 AI 交易策略的決策過程</p>
      </div>

      {/* 參數設置 */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <span className="text-lg">📊</span>
            回測參數設置
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
            <div>
              <Label htmlFor="symbol">股票代碼</Label>
              <Input
                id="symbol"
                value={symbol}
                onChange={(e) => setSymbol(e.target.value.toUpperCase())}
                placeholder="例如: AAPL"
                disabled={isRunning}
              />
            </div>
            
            <div>
              <Label htmlFor="period">回測期間</Label>
              <Select value={period} onValueChange={setPeriod} disabled={isRunning}>
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="6mo">6個月</SelectItem>
                  <SelectItem value="1y">1年</SelectItem>
                  <SelectItem value="2y">2年</SelectItem>
                  <SelectItem value="5y">5年</SelectItem>
                </SelectContent>
              </Select>
            </div>
          </div>
          
          <div className="bg-blue-50 p-4 rounded-lg mb-4">
            <div className="text-sm text-blue-800">
              <p className="font-medium">💰 資金模式：無上限資金</p>
              <p className="text-xs mt-1">系統使用無上限資金模式，所有損益計算基於實際交易成本，不依賴初始資金設定</p>
            </div>
          </div>
          
          <div className="flex gap-2 mt-4">
            <Button 
              onClick={startStreaming} 
              disabled={isRunning || isStarting} 
              className="flex-1"
            >
              {(isRunning || isStarting) ? (
                <>
                  <span className="mr-2 inline-block h-4 w-4 animate-spin rounded-full border-2 border-current border-t-transparent" />
                  {isStarting ? '正在啟動...' : '回測進行中...'}
                </>
              ) : (
                <>
                  <span className="mr-2">▶</span>
                  開始串流回測
                </>
              )}
            </Button>
            
            {(isRunning || isStarting) && (
              <Button onClick={stopStreaming} variant="destructive">
                <span className="mr-2">⏹</span>
                停止
              </Button>
            )}
          </div>
        </CardContent>
      </Card>

      {/* 進度顯示 */}
      {isRunning && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <span className="text-lg">⚡</span>
              即時進度與績效
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              <div>
                <div className="flex justify-between text-sm mb-2">
                  <span>處理進度</span>
                  <span>{progress.toFixed(1)}%</span>
                </div>
                <Progress value={progress} className="w-full" />
              </div>
              
              {/* 動態績效指標 */}
              <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                <div className="text-center p-2 bg-green-50 rounded">
                  <div className="text-lg font-bold text-green-600">
                    {dynamicPerformance.total_trades}
                  </div>
                  <div className="text-xs text-gray-600">已完成交易</div>
                </div>
                <div className="text-center p-2 bg-blue-50 rounded">
                  <div className="text-lg font-bold text-blue-600">
                    {(dynamicPerformance.win_rate * 100).toFixed(1)}%
                  </div>
                  <div className="text-xs text-gray-600">策略勝率</div>
                </div>
                <div className="text-center p-2 bg-purple-50 rounded">
                  <div className={`text-lg font-bold ${(dynamicPerformance.total_realized_pnl ?? 0) >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                    ${(dynamicPerformance.total_realized_pnl ?? 0).toFixed(2)}
                  </div>
                  <div className="text-xs text-gray-600">累積實現損益</div>
                </div>
                <div className="text-center p-2 bg-orange-50 rounded">
                  <div className={`text-lg font-bold ${(dynamicPerformance.cumulative_trade_return_rate ?? 0) >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                    {((dynamicPerformance.cumulative_trade_return_rate ?? 0) * 100).toFixed(2)}%
                  </div>
                  <div className="text-xs text-gray-600">累積交易收益率</div>
                </div>
              </div>
              
              {/* P&L 狀態顯示 */}
              {pnlStatus && (
                <div className="border rounded-lg p-4 bg-gradient-to-r from-green-50 to-blue-50">
                  <div className="text-sm font-semibold mb-3 flex items-center gap-2">
                    <span className="text-base">📈</span>
                    當前交易狀態
                  </div>
                  <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
                    <div className="text-center">
                      <div className={`text-xl font-bold ${(pnlStatus.unrealized_pnl ?? 0) >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                        ${(pnlStatus.unrealized_pnl ?? 0).toFixed(2)}
                      </div>
                      <div className="text-xs text-gray-600">未實現損益</div>
                    </div>
                    <div className="text-center">
                      <div className={`text-xl font-bold ${(pnlStatus.unrealized_pnl_pct ?? 0) >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                        {(pnlStatus.unrealized_pnl_pct ?? 0).toFixed(2)}%
                      </div>
                      <div className="text-xs text-gray-600">本次交易收益率</div>
                    </div>
                    <div className="text-center">
                      <div className="text-xl font-bold text-blue-600">
                        {pnlStatus.shares ? `${(pnlStatus.shares / 1000).toFixed(1)}k股` : '無持倉'}
                      </div>
                      <div className="text-xs text-gray-600">持股數量</div>
                    </div>
                  </div>
                  <div className="mt-3 text-xs text-gray-500 text-center">
                    風險等級: <span className={`font-semibold ${
                      pnlStatus.risk_level === 'high' ? 'text-red-600' : 
                      pnlStatus.risk_level === 'medium' ? 'text-yellow-600' : 'text-green-600'
                    }`}>{pnlStatus.risk_level ?? 'normal'}</span>
                  </div>
                </div>
              )}
              
              {currentStep && (
                <div className="p-3 bg-blue-50 rounded-lg">
                  <p className="text-sm text-blue-800">{currentStep}</p>
                </div>
              )}
            </div>
          </CardContent>
        </Card>
      )}

      {/* 即時日誌 */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <span className="text-lg">📈</span>
            即時決策日誌
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="h-96 overflow-y-auto bg-gray-50 p-4 rounded-lg space-y-2">
            {messages.map((message, index) => {
              // 根據訊息類型設定樣式
              let messageClass = "text-sm p-3 rounded-md leading-relaxed"
              
              if (message.includes('🤖') && message.includes('LLM決策')) {
                // LLM決策訊息 - 特殊樣式，更大空間顯示完整內容
                messageClass += " bg-gradient-to-r from-blue-50 to-indigo-50 border border-blue-200 text-blue-900"
              } else if (message.includes('🟢') || message.includes('🚀')) {
                // 買入相關訊息
                messageClass += " bg-green-100 border-l-4 border-green-500 text-green-800"
              } else if (message.includes('🔴') || message.includes('📤')) {
                // 賣出相關訊息  
                messageClass += " bg-red-100 border-l-4 border-red-500 text-red-800"
              } else if (message.includes('🟡')) {
                // 持有相關訊息
                messageClass += " bg-yellow-100 border-l-4 border-yellow-500 text-yellow-800"
              } else if (message.includes('💰')) {
                // 績效更新訊息
                messageClass += " bg-blue-100 border-l-4 border-blue-500 text-blue-800 font-semibold"
              } else if (message.includes('✅') || message.includes('完成')) {
                // 完成訊息
                messageClass += " bg-purple-100 border-l-4 border-purple-500 text-purple-800"
              } else {
                // 一般訊息
                messageClass += " bg-white border-l-4 border-gray-300 text-gray-700"
              }
              
              return (
                <div key={index} className={messageClass}>
                  <div className="flex items-start gap-2">
                    <span className="text-xs text-gray-500 min-w-fit">
                      [{new Date().toLocaleTimeString()}]
                    </span>
                    <span className="flex-1 whitespace-pre-wrap break-words">{message}</span>
                  </div>
                </div>
              )
            })}
            <div ref={messagesEndRef} />
          </div>
        </CardContent>
      </Card>

      {/* 錯誤顯示 */}
      {error && (
        <Alert variant="destructive">
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}

      {/* 結果顯示 */}
      {result && currentRunId && (
        <BacktestResultsWithAnalysis
          backtestResult={result}
          runId={currentRunId}
        />
      )}
    </div>
  )
}
