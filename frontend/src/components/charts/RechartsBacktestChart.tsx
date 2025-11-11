'use client'

import React from 'react'
import {
  ComposedChart,
  Line,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  Scatter,
  ReferenceDot
} from 'recharts'
import { StockData, TradingSignal, LLMDecisionLog } from '@/types'

interface RechartsBacktestChartProps {
  /** Stock price data */
  stockData: StockData[]
  /** Trading signal data */
  signals?: TradingSignal[]
  /** LLM decision records */
  llmDecisions?: LLMDecisionLog[]
  /** Chart height */
  height?: number
}

/**
 * Backtest Result Chart Component using Recharts
 * Alternative to BacktestChart with better React compatibility
 */
export function RechartsBacktestChart({
  stockData,
  signals = [],
  llmDecisions = [],
  height = 500,
}: RechartsBacktestChartProps) {
  // Validate and filter data
  const validStockData = React.useMemo(() => {
    if (!stockData || !Array.isArray(stockData)) {
      return []
    }
    
    return stockData.filter(item => {
      if (!item || typeof item !== 'object') return false
      if (!item.timestamp) return false
      if (typeof item.close !== 'number' || !isFinite(item.close)) return false
      return true
    })
  }, [stockData])

  // Prepare chart data
  const chartData = React.useMemo(() => {
    return validStockData.map((stock, index) => {
      // Find any signals for this date
      const signal = signals.find(s => s.timestamp === stock.timestamp)
      const llmDecision = llmDecisions.find(d => d.timestamp === stock.timestamp)
      
      return {
        date: stock.timestamp.split('T')[0], // Show date only
        close: stock.close,
        high: stock.high,
        low: stock.low,
        volume: stock.volume,
        buySignal: signal?.signal_type === 'BUY' ? stock.close : null,
        sellSignal: signal?.signal_type === 'SELL' ? stock.close : null,
        llmThinking: llmDecision ? stock.close : null,
      }
    })
  }, [validStockData, signals, llmDecisions])

  if (!validStockData.length) {
    return (
      <div className="w-full h-64 flex items-center justify-center bg-gray-50 rounded-lg">
        <p className="text-gray-500">No valid backtest data</p>
      </div>
    )
  }

  return (
    <div className="w-full">
      <ResponsiveContainer width="100%" height={height}>
        <ComposedChart
          data={chartData}
          margin={{ top: 20, right: 30, left: 20, bottom: 20 }}
        >
          <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
          <XAxis 
            dataKey="date" 
            tick={{ fontSize: 12 }}
            angle={-45}
            textAnchor="end"
            height={60}
          />
          <YAxis 
            yAxisId="price"
            tick={{ fontSize: 12 }}
            label={{ value: 'Price ($)', angle: -90, position: 'insideLeft' }}
          />
          <YAxis 
            yAxisId="volume"
            orientation="right"
            tick={{ fontSize: 12 }}
            label={{ value: 'Volume', angle: 90, position: 'insideRight' }}
          />
          <Tooltip 
            content={({ active, payload }) => {
              if (active && payload && payload.length) {
                const data = payload[0].payload
                return (
                  <div className="bg-white p-3 border border-gray-300 rounded-lg shadow-lg">
                    <p className="font-semibold text-gray-900">{data.date}</p>
                    <p className="text-sm text-gray-700">Close: ${data.close.toFixed(2)}</p>
                    <p className="text-sm text-gray-700">High: ${data.high.toFixed(2)}</p>
                    <p className="text-sm text-gray-700">Low: ${data.low.toFixed(2)}</p>
                    <p className="text-sm text-gray-600">Volume: {(data.volume / 1000000).toFixed(2)}M</p>
                    {data.buySignal && <p className="text-sm text-green-600 font-semibold">🚀 BUY Signal</p>}
                    {data.sellSignal && <p className="text-sm text-red-600 font-semibold">📤 SELL Signal</p>}
                    {data.llmThinking && <p className="text-sm text-yellow-600">🤖 AI Thinking</p>}
                  </div>
                )
              }
              return null
            }}
          />
          <Legend />
          
          {/* Price line */}
          <Line 
            yAxisId="price"
            type="monotone" 
            dataKey="close" 
            stroke="#2563eb" 
            strokeWidth={2}
            dot={false}
            name="Close Price"
          />
          
          {/* Volume bars */}
          <Area
            yAxisId="volume"
            type="monotone"
            dataKey="volume"
            fill="#94a3b8"
            stroke="#64748b"
            fillOpacity={0.3}
            name="Volume"
          />
          
          {/* Buy signals */}
          <Scatter
            yAxisId="price"
            dataKey="buySignal"
            fill="#10b981"
            shape="triangle"
            name="Buy Signal"
          />
          
          {/* Sell signals */}
          <Scatter
            yAxisId="price"
            dataKey="sellSignal"
            fill="#ef4444"
            shape="triangleDown"
            name="Sell Signal"
          />
          
          {/* LLM thinking points */}
          <Scatter
            yAxisId="price"
            dataKey="llmThinking"
            fill="#eab308"
            shape="circle"
            name="AI Decision"
          />
        </ComposedChart>
      </ResponsiveContainer>
      
      {/* Legend */}
      <div className="flex flex-wrap justify-center mt-4 space-x-4 text-sm">
        <div className="flex items-center space-x-2">
          <div className="w-3 h-1 bg-blue-600"></div>
          <span>Close Price</span>
        </div>
        
        <div className="flex items-center space-x-2">
          <div className="w-3 h-3 bg-gray-400 rounded"></div>
          <span>Volume</span>
        </div>
        
        {signals.length > 0 && (
          <>
            <div className="flex items-center space-x-2">
              <span className="text-green-600 text-lg font-bold">▲</span>
              <span className="font-medium">Buy Signal</span>
            </div>
            <div className="flex items-center space-x-2">
              <span className="text-red-500 text-lg font-bold">▼</span>
              <span className="font-medium">Sell Signal</span>
            </div>
          </>
        )}
        
        {llmDecisions.length > 0 && (
          <div className="flex items-center space-x-2">
            <div className="w-3 h-3 bg-yellow-500 rounded-full"></div>
            <span>AI Decision</span>
          </div>
        )}
      </div>
      
      {/* Statistics */}
      {(signals.length > 0 || llmDecisions.length > 0) && (
        <div className="mt-4 p-3 bg-gray-50 rounded-lg">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
            {signals.length > 0 && (
              <>
                <div>
                  <span className="text-gray-600">Trading Signals:</span>
                  <span className="ml-2 font-medium">{signals.length}</span>
                </div>
                <div>
                  <span className="text-gray-600">Buy:</span>
                  <span className="ml-2 font-medium text-green-600 text-lg">
                    ▲ {signals.filter(s => s.signal_type === 'BUY').length}
                  </span>
                </div>
                <div>
                  <span className="text-gray-600">Sell:</span>
                  <span className="ml-2 font-medium text-red-500 text-lg">
                    ▼ {signals.filter(s => s.signal_type === 'SELL').length}
                  </span>
                </div>
              </>
            )}
            {llmDecisions.length > 0 && (
              <div>
                <span className="text-gray-600">AI Decisions:</span>
                <span className="ml-2 font-medium">{llmDecisions.length}</span>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  )
}

