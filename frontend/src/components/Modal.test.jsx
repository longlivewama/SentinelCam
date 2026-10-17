import { fireEvent, render, screen } from '@testing-library/react'
import { useState } from 'react'
import { describe, expect, it, vi } from 'vitest'
import BarChart from './BarChart'
import Modal from './Modal'

describe('Modal', () => {
  it('is announced as a modal dialog named by its title', () => {
    render(<Modal title="Edit camera" onClose={() => {}}>body</Modal>)
    const dialog = screen.getByRole('dialog', { name: 'Edit camera' })
    expect(dialog).toHaveAttribute('aria-modal', 'true')
  })

  it('closes on Escape', () => {
    const onClose = vi.fn()
    render(<Modal title="Results" onClose={onClose}>body</Modal>)
    fireEvent.keyDown(document, { key: 'Escape' })
    expect(onClose).toHaveBeenCalledTimes(1)
  })

  it('does not close on a backdrop click, which would discard a form', () => {
    const onClose = vi.fn()
    render(<Modal title="Add camera" onClose={onClose}>body</Modal>)
    fireEvent.click(screen.getByRole('dialog').parentElement)
    expect(onClose).not.toHaveBeenCalled()
  })

  it('keeps focus in a field across re-renders', () => {
    // Callers pass onClose inline, so every render hands the dialog a new
    // function; that must not re-run its focus handling.
    function Form() {
      const [value, setValue] = useState('')
      return (
        <Modal title="Add camera" onClose={() => {}}>
          <input aria-label="Name" value={value} onChange={(e) => setValue(e.target.value)} />
        </Modal>
      )
    }
    render(<Form />)
    const input = screen.getByLabelText('Name')
    input.focus()
    fireEvent.change(input, { target: { value: 'Lobby' } })
    expect(document.activeElement).toBe(input)
  })
})

describe('BarChart', () => {
  it('says nothing happened instead of drawing an empty baseline', () => {
    const data = [{ label: '10-01', value: 0 }, { label: '10-02', value: 0 }]
    render(<BarChart data={data} emptyLabel="No falls detected in this window." />)
    expect(screen.getByText('No falls detected in this window.')).toBeInTheDocument()
  })

  it('draws bars when any value is non-zero', () => {
    const data = [{ label: '10-01', value: 0 }, { label: '10-02', value: 3 }]
    const { container } = render(<BarChart data={data} emptyLabel="Nothing" />)
    expect(screen.queryByText('Nothing')).not.toBeInTheDocument()
    expect(container.querySelectorAll('svg rect').length).toBeGreaterThan(0)
  })
})
