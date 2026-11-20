import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "CyberSLM Lab",
  description: "Fault-tolerant cybersecurity evidence and model demonstration",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
