import {
  HeadContent,
  Scripts,
  createRootRoute,
  Outlet,
} from '@tanstack/react-router'
import { TanStackRouterDevtools } from '@tanstack/react-router-devtools'
import * as React from 'react'
import { DefaultCatchBoundary } from '~/components/DefaultCatchBoundary'
import { NotFound } from '~/components/NotFound'
import appCss from '~/styles/app.css?url'
import { seo } from '~/utils/seo'
import { scan } from 'react-scan'

export const Route = createRootRoute({
  head: () => ({
    meta: [
      {
        charSet: 'utf-8',
      },
      {
        name: 'viewport',
        content: 'width=device-width, initial-scale=1',
      },
      ...seo({
        title: 'XAI Platform - Audio & Image Analysis',
        description: `Explainable AI platform for deepfake audio detection and lung cancer analysis with multiple XAI techniques.`,
      }),
    ],
    links: [
      { rel: 'stylesheet', href: appCss },
    ],
  }),
  errorComponent: DefaultCatchBoundary,
  notFoundComponent: () => <NotFound />,
  component: RootLayout,
})

function RootLayout() {
  React.useEffect(() => {
    if (typeof window !== 'undefined') {
      scan()
    }
  }, [])

  return (
    <html className="h-full">
      <head>
        <HeadContent />
      </head>
      <body className="min-h-screen flex flex-col bg-gradient-to-b from-slate-50 to-slate-100 dark:from-slate-950 dark:to-slate-900">
        <header className="border-b border-slate-200 dark:border-slate-800 flex-shrink-0">
          <div className="px-6 py-4 max-w-7xl mx-auto">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-3">
                <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-blue-500 to-purple-600 flex items-center justify-center text-white font-bold">
                  XAI
                </div>
                <div>
                  <h1 className="text-2xl font-bold text-slate-900 dark:text-slate-50">XAI Platform</h1>
                  <p className="text-sm text-slate-600 dark:text-slate-400">Explainable AI for Audio & Image Analysis</p>
                </div>
              </div>
            </div>
          </div>
        </header>
        <main className="flex-1 overflow-auto">
          <Outlet />
        </main>
        <TanStackRouterDevtools position="bottom-right" />
        <Scripts />
      </body>
    </html>
  )
}
