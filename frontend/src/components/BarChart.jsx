import { useState } from 'react'

/**
 * Minimal single-series bar chart (magnitude over category/time). Thin
 * bars with rounded data-ends, recessive gridlines, and a hover tooltip.
 * No legend - a single series is named by the chart title, not a legend
 * box (see dataviz skill guidance).
 */
export default function BarChart({ data, height = 160, valueFormatter = (v) => v, emptyLabel = 'No data yet' }) {
  const [hovered, setHovered] = useState(null)

  if (!data || data.length === 0) {
    return (
      <div className="flex items-center justify-center text-sm text-slate-500" style={{ height }}>
        {emptyLabel}
      </div>
    )
  }

  const max = Math.max(...data.map((d) => d.value), 1)
  const barCount = data.length
  const gap = 6
  const chartWidth = Math.max(barCount * 28, 200)
  const barWidth = (chartWidth - gap * (barCount - 1)) / barCount
  const topPadding = 24

  return (
    // Reserved top padding (pt-9) gives the hover tooltip room to render
    // above the bars without being clipped: this outer element sets no
    // overflow itself (a mix of overflow-x:auto/overflow-y:visible on the
    // SAME element computes the visible axis to auto too, per the CSS
    // Overflow spec - so the scrollable wrapper below is a separate,
    // inner element instead).
    <div className="relative w-full pt-9">
      {hovered !== null && (
        <div
          className="pointer-events-none absolute top-0 -translate-x-1/2 rounded-md border border-surface-600 bg-surface-800 px-2 py-1 text-xs text-slate-200 shadow-lg"
          style={{
            left: `${((hovered * (barWidth + gap) + barWidth / 2) / chartWidth) * 100}%`,
          }}
        >
          <div className="font-semibold">{valueFormatter(data[hovered].value)}</div>
          <div className="text-slate-400">{data[hovered].label}</div>
        </div>
      )}
      <div className="overflow-x-auto">
        <svg
          viewBox={`0 0 ${chartWidth} ${height}`}
          width="100%"
          height={height}
          preserveAspectRatio="none"
          className="overflow-visible"
        >
          {/* recessive baseline */}
          <line x1={0} y1={height - 1} x2={chartWidth} y2={height - 1} stroke="#1c242f" strokeWidth={1} />
          {data.map((d, i) => {
            const barHeight = Math.max(((d.value / max) * (height - topPadding)), d.value > 0 ? 3 : 0)
            const x = i * (barWidth + gap)
            const y = height - barHeight
            const isHovered = hovered === i
            return (
              <g
                key={d.label + i}
                onMouseEnter={() => setHovered(i)}
                onMouseLeave={() => setHovered(null)}
                className="cursor-pointer"
              >
                <rect x={x} y={0} width={barWidth} height={height} fill="transparent" />
                <rect
                  x={x}
                  y={y}
                  width={barWidth}
                  height={barHeight}
                  rx={4}
                  fill={isHovered ? '#22d3ee' : '#22d3ee99'}
                />
              </g>
            )
          })}
        </svg>
        <div className="mt-1 flex text-[10px] text-slate-500" style={{ width: chartWidth }}>
          {data.map((d, i) => (
            <div key={d.label + i} style={{ width: barWidth + gap }} className="truncate text-center">
              {i % Math.ceil(barCount / 10 || 1) === 0 ? d.label : ''}
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}
