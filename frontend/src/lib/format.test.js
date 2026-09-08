import { describe, expect, it } from 'vitest'
import { detectorMeta, eventTypeMeta, formatVideoTimestamp } from './format'

describe('formatVideoTimestamp', () => {
  it('renders a position within a video as m:ss', () => {
    expect(formatVideoTimestamp(0)).toBe('0:00')
    expect(formatVideoTimestamp(12.5)).toBe('0:12')
    expect(formatVideoTimestamp(65)).toBe('1:05')
    expect(formatVideoTimestamp(245)).toBe('4:05')
  })

  it('adds an hours component past an hour', () => {
    expect(formatVideoTimestamp(3600)).toBe('1:00:00')
    expect(formatVideoTimestamp(3723)).toBe('1:02:03')
  })

  it('returns null for a missing position so callers can say so explicitly', () => {
    // Deliberately not '—': the caller needs to distinguish "no position
    // recorded" from a real 0:00, and render different text for it.
    expect(formatVideoTimestamp(null)).toBeNull()
    expect(formatVideoTimestamp(undefined)).toBeNull()
    expect(formatVideoTimestamp(NaN)).toBeNull()
  })

  it('clamps a negative position rather than rendering a negative time', () => {
    expect(formatVideoTimestamp(-5)).toBe('0:00')
  })
})

describe('eventTypeMeta', () => {
  it('gives the product’s primary event type a proper label', () => {
    // This previously fell through to the generic branch and rendered as a
    // bare lowercase "fall" with a bell icon.
    expect(eventTypeMeta('fall').label).toBe('Fall Detected')
    expect(eventTypeMeta('violence').label).toBe('Violence Detected')
  })

  it('still falls back gracefully for an unknown type', () => {
    expect(eventTypeMeta('something_new').label).toBe('something_new')
    expect(eventTypeMeta(null).label).toBe('Unknown')
  })
})

describe('detectorMeta', () => {
  it('names the strategy that produced an event', () => {
    expect(detectorMeta('model').label).toBe('AI model')
    expect(detectorMeta('heuristic').label).toBe('Pose heuristic')
  })

  it('returns null for rows recorded before the detector was tracked', () => {
    expect(detectorMeta(null)).toBeNull()
    expect(detectorMeta('unknown')).toBeNull()
  })
})
