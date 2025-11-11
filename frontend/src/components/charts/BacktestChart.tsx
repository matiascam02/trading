'use client'

import React, { useEffect, useRef } from 'react'
import { createChart, Time } from 'lightweight-charts'
import { StockData, TradingSignal, LLMDecisionLog } from '@/types'

interface BacktestChartProps {
  /** 股票價格數據 */
  stockData: StockData[]
  /** 交易信號數據 */
  signals?: TradingSignal[]
  /** LLM 決策記錄 */
  llmDecisions?: LLMDecisionLog[]
  /** 圖表高度 */
  height?: number
  /** 是否顯示成交量 */
  showVolume?: boolean
}

/**
 * 回測結果圖表組件
 * 專注於顯示交易信號和 LLM 決策，使用 TradingView Lightweight Charts
 */
export function BacktestChart({
  stockData,
  signals = [],
  llmDecisions = [],
  height = 500,
  showVolume = true,
}: BacktestChartProps) {
  const chartContainerRef = useRef<HTMLDivElement>(null)
  const chartInstanceRef = useRef<any>(null)

  // 數據驗證和過濾
  const validStockData = React.useMemo(() => {
    if (!stockData || !Array.isArray(stockData)) {
      return []
    }
    
    return stockData.filter(item => {
      if (!item || typeof item !== 'object') return false
      if (!item.timestamp) return false
      if (typeof item.open !== 'number' || !isFinite(item.open)) return false
      if (typeof item.high !== 'number' || !isFinite(item.high)) return false
      if (typeof item.low !== 'number' || !isFinite(item.low)) return false
      if (typeof item.close !== 'number' || !isFinite(item.close)) return false
      if (typeof item.volume !== 'number' || !isFinite(item.volume) || item.volume < 0) return false
      
      // OHLC 邏輯驗證
      if (item.high < item.low || item.high < item.open || item.high < item.close) return false
      if (item.low > item.open || item.low > item.close) return false
      
      return true
    })
  }, [stockData])

  // 時間轉換函數
  const convertTimestamp = (timestamp: string): number => {
    let date: Date
    
    if (timestamp.includes('T')) {
      date = new Date(timestamp)
    } else if (timestamp.includes('-')) {
      date = new Date(timestamp + 'T00:00:00.000Z')
    } else {
      date = new Date(timestamp)
    }
    
    if (isNaN(date.getTime())) {
      console.warn('無效的時間格式:', timestamp)
      return Math.floor(Date.now() / 1000)
    }
    
    return Math.floor(date.getTime() / 1000)
  }

  useEffect(() => {
    if (!chartContainerRef.current || !validStockData.length) {
      return
    }

    // Add a small delay to ensure DOM is fully settled (React 19 compatibility)
    const timeoutId = setTimeout(() => {
      if (!chartContainerRef.current) return

      // 調試信息 - 檢查傳入的數據
      console.log('BacktestChart 數據調試:', {
        stockDataLength: validStockData.length,
        signalsLength: signals.length,
        llmDecisionsLength: llmDecisions.length,
        firstSignal: signals[0],
        firstLLMDecision: llmDecisions[0],
        stockDataSample: validStockData.slice(0, 2)
      })

      // 使用現有的容器節點，避免在嚴格模式下反覆插入/移除造成的 DOM 錯誤
      const container = chartContainerRef.current
      container.style.height = `${height}px`

      // 創建圖表
      const chart = createChart(container, {
        width: container.clientWidth,
        height: height,
        layout: {
          backgroundColor: '#ffffff',
          textColor: '#333',
        },
        grid: {
          vertLines: { color: '#f0f0f0' },
          horzLines: { color: '#f0f0f0' },
        },
        rightPriceScale: {
          borderColor: '#cccccc',
        },
        timeScale: {
          borderColor: '#cccccc',
          timeVisible: true,
          secondsVisible: false,
        },
      })
      
      // Store chart instance for cleanup
      chartInstanceRef.current = chart

    // K線數據
    const candlestickData = validStockData.map(stock => ({
      time: convertTimestamp(stock.timestamp) as Time,
      open: stock.open,
      high: stock.high,
      low: stock.low,
      close: stock.close,
    }))

    // 添加K線圖
    const candlestickSeries = chart.addCandlestickSeries({
      upColor: '#26a69a',
      downColor: '#ef5350',
      borderDownColor: '#ef5350',
      borderUpColor: '#26a69a',
      wickDownColor: '#ef5350',
      wickUpColor: '#26a69a',
    })
    candlestickSeries.setData(candlestickData)

    // 添加成交量
    if (showVolume) {
      const volumeData = validStockData.map(stock => ({
        time: convertTimestamp(stock.timestamp) as Time,
        value: stock.volume,
        color: stock.close >= stock.open ? '#26a69a' : '#ef5350',
      }))

      const volumeSeries = chart.addHistogramSeries({
        color: '#26a69a',
        priceFormat: {
          type: 'volume',
        },
        priceScaleId: 'volume',
      })
      
      volumeSeries.setData(volumeData)

      chart.priceScale('volume').applyOptions({
        scaleMargins: {
          top: 0.8,
          bottom: 0,
        },
      })
    }

    // 合併並處理所有標記（交易信號 + LLM決策）
    const allMarkers: any[] = []

    // 1. 添加交易信號標記 (BUY/SELL)
    if (signals.length > 0) {
      console.log('處理交易信號數據:', signals)
      
      const validSignals = signals.filter(signal => {
        const isValid = signal && signal.timestamp && signal.signal_type && 
               typeof signal.price === 'number' && isFinite(signal.price)
        
        if (!isValid) {
          console.warn('無效的信號數據:', signal)
        }
        return isValid
      })

      console.log(`有效信號數量: ${validSignals.length}/${signals.length}`)

      const tradingMarkers = validSignals.map(signal => {
        const marker = {
          time: convertTimestamp(signal.timestamp) as Time,
          position: (signal.signal_type === 'BUY' ? 'belowBar' : 'aboveBar') as 'belowBar' | 'aboveBar',
          color: signal.signal_type === 'BUY' ? '#26a69a' : '#ef5350',
          shape: (signal.signal_type === 'BUY' ? 'arrowUp' : 'arrowDown') as 'arrowUp' | 'arrowDown',
          text: signal.signal_type === 'BUY' ? 'BUY' : 'SELL',
          size: 2, // 調整箭頭大小 (預設是1，範圍0-4)
          id: `signal_${signal.timestamp}_${signal.signal_type}` // 防止重複
        }
        console.log('創建交易標記:', { original: signal, marker })
        return marker
      })
      
      allMarkers.push(...tradingMarkers)
    }

    // 2. 添加 LLM 純思考決策標記（不包含實際交易的決策）
    if (llmDecisions.length > 0) {
      console.log('處理LLM決策數據:', llmDecisions)
      
      const validDecisions = llmDecisions.filter(decision => {
        // 檢查基本數據結構：需要 timestamp 和 reasoning
        const hasBasicData = decision && decision.timestamp && decision.reasoning
        
        // LLM 決策應該是 action: "THINK"，不是實際的交易信號
        const isThinkingDecision = decision.action === 'THINK'
        
        const isValid = hasBasicData && isThinkingDecision
        if (!isValid) {
          console.warn('過濾掉的LLM決策:', decision, { 
            hasBasicData, 
            isThinkingDecision, 
            actualAction: decision.action 
          })
        }
        return isValid
      })

      console.log(`有效LLM決策數量: ${validDecisions.length}/${llmDecisions.length}`)

      const thinkingMarkers = validDecisions.map(decision => {
        const confidence = decision.confidence || decision.decision?.confidence || 0.5
        const alpha = Math.max(0.6, confidence)
        
        // 使用 timestamp 字段（新格式）或 date 字段（向後兼容）
        const timeValue = decision.timestamp || decision.date || ''
        
        const marker = {
          time: convertTimestamp(timeValue) as Time,
          position: 'aboveBar' as 'aboveBar',  // 在K棒上方
          color: `rgba(255, 193, 7, ${alpha})`, // 黃色，根據信心度調整透明度
          shape: 'arrowDown' as 'arrowDown',   // 向下箭頭
          text: 'AI',  // 簡潔的AI標識
          size: 1.0,   // 適中的大小
          id: `llm_${timeValue}_thinking`
        }
        console.log('創建LLM標記:', { original: decision, marker })
        return marker
      })
      
      allMarkers.push(...thinkingMarkers)
    }

    // 3. 設置合併後的標記
    if (allMarkers.length > 0) {
      try {
        // 按時間排序標記
        allMarkers.sort((a, b) => (a.time as number) - (b.time as number))
        candlestickSeries.setMarkers(allMarkers)
        console.log(`✅ 成功設置了 ${allMarkers.length} 個圖表標記:`, allMarkers)
      } catch (error) {
        console.error('❌ 設置圖表標記時出錯:', error)
      }
    } else {
      console.log('⚠️ 沒有任何標記數據可設置')
      
      // 如果沒有真實數據，創建一些測試標記來驗證圖表功能
      if (validStockData.length > 10) {
        const testMarkers = [
          {
            time: convertTimestamp(validStockData[5].timestamp) as Time,
            position: 'belowBar' as 'belowBar',
            color: '#26a69a',
            shape: 'arrowUp' as 'arrowUp',
            text: 'B',
            size: 2,
          },
          {
            time: convertTimestamp(validStockData[10].timestamp) as Time,
            position: 'aboveBar' as 'aboveBar',
            color: '#ef5350',
            shape: 'arrowDown' as 'arrowDown',
            text: 'S',
            size: 2,
          },
          {
            time: convertTimestamp(validStockData[7].timestamp) as Time,
            position: 'inBar' as 'inBar',
            color: 'rgba(255, 193, 7, 0.8)',
            shape: 'circle' as 'circle',
            text: '💭',
            size: 1.2,
          }
        ]
        
        try {
          candlestickSeries.setMarkers(testMarkers)
          console.log('🧪 設置了測試標記來驗證圖表功能')
        } catch (error) {
          console.error('❌ 設置測試標記失敗:', error)
        }
      }
    }

    chart.timeScale().fitContent()

    // 響應式調整
    const handleResize = () => {
      if (container && container.isConnected) {
        chart.applyOptions({
          width: container.clientWidth,
        })
      }
    }

      window.addEventListener('resize', handleResize)
      
      // Return cleanup function for this chart instance
      return () => {
        window.removeEventListener('resize', handleResize)
        if (chartInstanceRef.current) {
          try {
            chartInstanceRef.current.remove()
            chartInstanceRef.current = null
          } catch {
            // Ignore cleanup errors
          }
        }
      }
    }, 0) // Small delay to let React finish DOM operations

    return () => {
      clearTimeout(timeoutId)
      // Also cleanup chart if timeout hasn't run yet
      if (chartInstanceRef.current) {
        try {
          chartInstanceRef.current.remove()
          chartInstanceRef.current = null
        } catch {
          // Ignore cleanup errors
        }
      }
    }
  }, [validStockData, signals, llmDecisions, height, showVolume])

  if (!validStockData.length) {
    return (
      <div className="w-full h-64 flex items-center justify-center bg-gray-50 rounded-lg">
        <p className="text-gray-500">無有效的回測數據</p>
      </div>
    )
  }

  return (
    <div className="w-full">
      <div 
        ref={chartContainerRef} 
        className="w-full border rounded-lg"
        style={{ height: `${height}px` }}
      />
      
      {/* 圖例 */}
      <div className="flex flex-wrap justify-center mt-4 space-x-4 text-sm">
        <div className="flex items-center space-x-2">
          <div className="flex space-x-1">
            <div className="w-2 h-4 bg-green-600"></div>
            <div className="w-2 h-4 bg-red-500"></div>
          </div>
          <span>K線圖</span>
        </div>
        
        {showVolume && (
          <div className="flex items-center space-x-2">
            <div className="w-3 h-3 bg-gray-400 rounded"></div>
            <span>成交量</span>
          </div>
        )}
        
        {signals.length > 0 && (
          <>
            <div className="flex items-center space-x-2">
              <span className="text-green-600 text-lg font-bold">▲</span>
              <span className="font-medium">買入信號</span>
            </div>
            <div className="flex items-center space-x-2">
              <span className="text-red-500 text-lg font-bold">▼</span>
              <span className="font-medium">賣出信號</span>
            </div>
          </>
        )}
        
        {llmDecisions.length > 0 && (
          <div className="flex items-center space-x-2">
            <div className="w-3 h-3 bg-yellow-500 rounded-full"></div>
            <span>AI 決策點</span>
          </div>
        )}
      </div>
      
      {/* 統計信息 */}
      {(signals.length > 0 || llmDecisions.length > 0) && (
        <div className="mt-4 p-3 bg-gray-50 rounded-lg">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
            {signals.length > 0 && (
              <>
                <div>
                  <span className="text-gray-600">交易信號:</span>
                  <span className="ml-2 font-medium">{signals.length}</span>
                </div>
                <div>
                  <span className="text-gray-600">買入:</span>
                  <span className="ml-2 font-medium text-green-600 text-lg">
                    ▲ {signals.filter(s => s.signal_type === 'BUY').length}
                  </span>
                </div>
                <div>
                  <span className="text-gray-600">賣出:</span>
                  <span className="ml-2 font-medium text-red-500 text-lg">
                    ▼ {signals.filter(s => s.signal_type === 'SELL').length}
                  </span>
                </div>
              </>
            )}
            {llmDecisions.length > 0 && (
              <div>
                <span className="text-gray-600">AI 決策:</span>
                <span className="ml-2 font-medium">{llmDecisions.length}</span>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  )
}
