declare module '*.css' {
  const content: { [className: string]: string }
  export default content
}

declare module '*.css?url' {
  const content: string
  export default content
}

// Image and asset module declarations so TypeScript can import them as strings
declare module '*.png' {
  const src: string
  export default src
}

declare module '*.jpg' {
  const src: string
  export default src
}

declare module '*.jpeg' {
  const src: string
  export default src
}

declare module '*.webp' {
  const src: string
  export default src
}

// SVGs may be imported either as URL (string) or as React component depending on setup.
// This generic declaration treats them as string URLs. If you use SVGR to import as React
// components, add a separate declaration (e.g. `declare module '*.svg' { import * as React from 'react'; export const ReactComponent: React.FC<React.SVGProps<SVGSVGElement>>; const src: string; export default src }`).
declare module '*.svg' {
  const src: string
  export default src
}

/// <reference types="vite/client" />