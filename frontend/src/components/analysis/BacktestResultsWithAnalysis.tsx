'use client'

import React from 'react'
import dynamic from 'next/dynamic'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Alert, AlertDescription } from '@/components/ui/alert'
import { Calendar, BarChart3, Brain, Info } from 'lucide-react'
import { BacktestResult } from '@/types'
import DayAnalysis from './DayAnalysis'

// Dynamically import BacktestChart to avoid SSR issues with lightweight-charts
const BacktestChart = dynamic(
  () => import('@/components/charts/BacktestChart').then(mod => ({ default: mod.BacktestChart })),
  { ssr: false, loading: () => <div className="h-[500px] flex items-center justify-center">Loading chart...</div> }
)

interface BacktestResultsWithAnalysisProps {
  backtestResult: BacktestResult
  runId: string
}

/**
 * Enhanced Backtest Results Component with Day Analysis
 * Combines traditional backtest results with day-by-day LLM analysis
 */
export default function BacktestResultsWithAnalysis({
  backtestResult,
  runId
}: BacktestResultsWithAnalysisProps) {
  // Handle date selection from day analysis
  const handleDateSelect = (date: string) => {
    console.log('Selected date for analysis:', date)
    // Additional logic can be added here if needed
  }

  // Format number with commas
  const formatNumber = (num: number): string => {
    return num.toLocaleString('en-US', { 
      minimumFractionDigits: 2, 
      maximumFractionDigits: 2 
    })
  }

  // Format percentage - backend already returns percentages, so no need to multiply by 100
  const formatPercentage = (num: number): string => {
    return `${num.toFixed(2)}%`
  }

  if (!backtestResult) {
    return (
      <Alert>
        <Info className="h-4 w-4" />
        <AlertDescription>
          No backtest result data available
        </AlertDescription>
      </Alert>
    )
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <BarChart3 className="h-5 w-5" />
            Backtest Results & Analysis
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="text-center">
              <div className="text-2xl font-bold text-blue-600">
                {backtestResult.statistics?.total_trades || 0}
              </div>
              <div className="text-sm text-gray-600">Total Trades</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-green-600">
                {formatPercentage(backtestResult.statistics?.win_rate || 0)}
              </div>
              <div className="text-sm text-gray-600">Win Rate</div>
            </div>
            <div className="text-center">
              <div className={`text-2xl font-bold ${
                (backtestResult.statistics?.total_return || 0) >= 0 ? 'text-green-600' : 'text-red-600'
              }`}>
                {formatPercentage(backtestResult.statistics?.total_return || 0)}
              </div>
              <div className="text-sm text-gray-600">Total Return</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-red-600">
                {formatPercentage(Math.abs(backtestResult.statistics?.max_drawdown || 0))}
              </div>
              <div className="text-sm text-gray-600">Max Drawdown</div>
            </div>
          </div>
          
          {/* Additional Statistics */}
          {backtestResult.statistics?.total_realized_pnl && (
            <div className="mt-4 pt-4 border-t">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="text-center">
                  <div className={`text-lg font-bold ${
                    backtestResult.statistics.total_realized_pnl >= 0 ? 'text-green-600' : 'text-red-600'
                  }`}>
                    ${formatNumber(backtestResult.statistics.total_realized_pnl)}
                  </div>
                  <div className="text-sm text-gray-600">Realized P&L</div>
                </div>
                {backtestResult.statistics.cumulative_trade_return_rate && (
                  <div className="text-center">
                    <div className="text-lg font-bold text-orange-600">
                      {formatPercentage(backtestResult.statistics.cumulative_trade_return_rate)}
                    </div>
                    <div className="text-sm text-gray-600">Cumulative Trade Return</div>
                  </div>
                )}
              </div>
            </div>
          )}
        </CardContent>
      </Card>

      {/* Data Summary */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Info className="h-5 w-5" />
            Data Overview
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div>
              <div className="text-sm text-gray-600">Stock Data Points</div>
              <div className="text-lg font-bold">
                {backtestResult.stock_data?.length || 0}
              </div>
            </div>
            <div>
              <div className="text-sm text-gray-600">Trading Signals</div>
              <div className="text-lg font-bold">
                {backtestResult.signals?.length || 0}
              </div>
            </div>
            <div>
              <div className="text-sm text-gray-600">LLM Decisions</div>
              <div className="text-lg font-bold">
                {backtestResult.llm_decisions?.length || 0}
              </div>
            </div>
            <div>
              <div className="text-sm text-gray-600">Analysis Features</div>
              <div className="text-lg font-bold">
                Daily Analysis
              </div>
            </div>
          </div>
          
          {/* Date Range Info */}
          <div className="mt-4 pt-4 border-t">
            <div className="flex items-center justify-center">
              <Badge variant="outline" className="text-center">
                <Calendar className="h-3 w-3 mr-1" />
                Use the daily analysis feature below to select a date for detailed analysis
              </Badge>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Price Chart Section */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <BarChart3 className="h-5 w-5" />
            Price Trend & Trading Signals
          </CardTitle>
        </CardHeader>
        <CardContent>
          <BacktestChart 
            stockData={backtestResult.stock_data || []}
            signals={backtestResult.signals || []}
            llmDecisions={backtestResult.llm_decisions || []}
            height={500}
            showVolume={true}
          />
        </CardContent>
      </Card>

      {/* Day Analysis Section */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Brain className="h-5 w-5" />
            Daily LLM Analysis
          </CardTitle>
        </CardHeader>
        <CardContent>
          <DayAnalysis
            runId={runId}
            onDateSelect={handleDateSelect}
          />
        </CardContent>
      </Card>
    </div>
  )
}
