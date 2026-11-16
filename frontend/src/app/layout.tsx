import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "UAV Sentinel",
  description: "Agentic Analysis of ARJUNA flight logs",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
