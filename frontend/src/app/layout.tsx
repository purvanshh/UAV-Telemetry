import type { Metadata, Viewport } from "next";
import { Inter, JetBrains_Mono } from "next/font/google";
import { AppHeader } from "@/components/AppHeader";
import { AmbientDrones } from "@/components/drone/AmbientDrones";
import { introScript, WelcomeIntro } from "@/components/drone/WelcomeIntro";
import { ToastProvider } from "@/components/Toast";
import "./globals.css";

const inter = Inter({ subsets: ["latin"], variable: "--font-inter", display: "swap" });
const mono = JetBrains_Mono({ subsets: ["latin"], variable: "--font-jetbrains", display: "swap" });

export const metadata: Metadata = {
  title: { default: "ARJUNA", template: "%s · ARJUNA" },
  description: "UAV flight analysis: upload a drone flight log and see what happened, what to check, and whether the drone is ready for its next mission.",
};

export const viewport: Viewport = {
  themeColor: [
    { media: "(prefers-color-scheme: light)", color: "#ffffff" },
    { media: "(prefers-color-scheme: dark)", color: "#0b0f14" },
  ],
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en" className={`${inter.variable} ${mono.variable}`} suppressHydrationWarning>
      <head>
        {/* decides before first paint whether the welcome plays, so the app never flashes first */}
        <script dangerouslySetInnerHTML={{ __html: introScript }} />
      </head>
      <body>
        <ToastProvider>
          <WelcomeIntro />
          <AmbientDrones />
          <div className="app-shell">
            <AppHeader />
            {children}
          </div>
        </ToastProvider>
      </body>
    </html>
  );
}
