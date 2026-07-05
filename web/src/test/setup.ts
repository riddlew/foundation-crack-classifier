import '@testing-library/jest-dom/vitest'

// jsdom does not implement the object-URL APIs the image store relies on.
let objectUrlCounter = 0
URL.createObjectURL = () => `blob:mock/${objectUrlCounter++}`
URL.revokeObjectURL = () => {}
