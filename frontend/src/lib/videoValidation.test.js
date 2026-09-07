import { describe, expect, it } from 'vitest'
import { MAX_UPLOAD_SIZE_MB, fileExtension, validateVideoFile } from './videoValidation'

function fakeFile(name, size = 1024) {
  return { name, size }
}

describe('fileExtension', () => {
  it('lowercases and keeps only the final extension', () => {
    expect(fileExtension('Clip.MP4')).toBe('.mp4')
    expect(fileExtension('my.video.backup.mov')).toBe('.mov')
  })

  it('returns an empty string when there is no extension', () => {
    expect(fileExtension('README')).toBe('')
    expect(fileExtension('')).toBe('')
  })
})

describe('validateVideoFile', () => {
  it('accepts every format the backend allows, case-insensitively', () => {
    for (const name of ['a.mp4', 'b.MOV', 'c.avi', 'd.mkv', 'e.webm']) {
      expect(validateVideoFile(fakeFile(name))).toBeNull()
    }
  })

  it('rejects a file whose extension is not on the allowlist', () => {
    expect(validateVideoFile(fakeFile('notes.txt'))).toMatch(/supported video formats/i)
  })

  it('rejects a file with no extension at all', () => {
    expect(validateVideoFile(fakeFile('README'))).toMatch(/no file extension/i)
  })

  it('rejects an empty file', () => {
    expect(validateVideoFile(fakeFile('empty.mp4', 0))).toMatch(/empty/i)
  })

  it('rejects a file over the size limit but accepts one exactly at it', () => {
    const limit = MAX_UPLOAD_SIZE_MB * 1024 * 1024
    expect(validateVideoFile(fakeFile('big.mp4', limit + 1))).toMatch(/over the 500MB limit/i)
    expect(validateVideoFile(fakeFile('exact.mp4', limit))).toBeNull()
  })

  it('treats a missing file as nothing to complain about', () => {
    expect(validateVideoFile(null)).toBeNull()
  })
})
