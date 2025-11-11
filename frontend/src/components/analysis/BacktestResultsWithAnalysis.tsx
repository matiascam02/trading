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
    return num.toLocaleString('zh-TW', { 
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
          沒有可用的回測結果數據
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
            回測結果與分析
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="text-center">
              <div className="text-2xl font-bold text-blue-600">
                {backtestResult.statistics?.total_trades || 0}
              </div>
              <div className="text-sm text-gray-600">總交易次數</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-green-600">
                {formatPercentage(backtestResult.statistics?.win_rate || 0)}
              </div>
              <div className="text-sm text-gray-600">勝率</div>
            </div>
            <div className="text-center">
              <div className={`text-2xl font-bold ${
                (backtestResult.statistics?.total_return || 0) >= 0 ? 'text-green-600' : 'text-red-600'
              }`}>
                {formatPercentage(backtestResult.statistics?.total_return || 0)}
              </div>
              <div className="text-sm text-gray-600">總回報率</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-red-600">
                {formatPercentage(Math.abs(backtestResult.statistics?.max_drawdown || 0))}
              </div>
              <div className="text-sm text-gray-600">最大回撤</div>
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
                  <div className="text-sm text-gray-600">已實現損益</div>
                </div>
                {backtestResult.statistics.cumulative_trade_return_rate && (
                  <div className="text-center">
                    <div className="text-lg font-bold text-orange-600">
                      {formatPercentage(backtestResult.statistics.cumulative_trade_return_rate)}
                    </div>
                    <div className="text-sm text-gray-600">累計交易回報率</div>
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
            數據概覽
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div>
              <div className="text-sm text-gray-600">股價數據點</div>
              <div className="text-lg font-bold">
                {backtestResult.stock_data?.length || 0}
              </div>
            </div>
            <div>
              <div className="text-sm text-gray-600">交易信號</div>
              <div className="text-lg font-bold">
                {backtestResult.signals?.length || 0}
              </div>
            </div>
            <div>
              <div className="text-sm text-gray-600">LLM 決策</div>
              <div className="text-lg font-bold">
                {backtestResult.llm_decisions?.length || 0}
              </div>
            </div>
            <div>
              <div className="text-sm text-gray-600">分析功能</div>
              <div className="text-lg font-bold">
                日別分析
              </div>
            </div>
          </div>
          
          {/* Date Range Info */}
          <div className="mt-4 pt-4 border-t">
            <div className="flex items-center justify-center">
              <Badge variant="outline" className="text-center">
                <Calendar className="h-3 w-3 mr-1" />
                請使用下方的日別分析功能選擇日期進行詳細分析
              </Badge>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Price Chart Section - Temporarily disabled due to lightweight-charts compatibility */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <BarChart3 className="h-5 w-5" />
            股價走勢與交易信號
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="h-[500px] flex items-center justify-center bg-gray-50 rounded-lg">
            <div className="text-center">
              <p className="text-gray-600 mb-2">Chart temporarily disabled</p>
              <p className="text-sm text-gray-500">Check stats above for results</p>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Day Analysis Section */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Brain className="h-5 w-5" />
            日別 LLM 分析
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
