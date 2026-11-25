/**
 * Optional static environment injector executed during Vercel build.
 * If RETINACARE_API_URL (or API_BASE_URL / VITE_API_URL / NEXT_PUBLIC_API_URL)
 * is defined in Vercel Project Environment Variables, writes it to js/env.js.
 */
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const apiUrl =
    process.env.RETINACARE_API_URL ||
    process.env.VITE_API_URL ||
    process.env.NEXT_PUBLIC_API_URL ||
    process.env.API_BASE_URL;

if (apiUrl && apiUrl.trim()) {
    const cleanUrl = apiUrl.trim().replace(/\/+$/, "");
    const envFilePath = path.join(__dirname, "js", "env.js");
    const content = [
        "/**",
        " * RetinaCare AI - Runtime Backend API Endpoint (Generated at Build)",
        " */",
        `window.__RETINACARE_API_URL__ = ${JSON.stringify(cleanUrl)};`,
        ""
    ].join("\n");
    fs.writeFileSync(envFilePath, content, "utf8");
    console.log(`[Vercel Build] Set backend API base URL -> ${cleanUrl}`);
} else {
    console.log("[Vercel Build] No RETINACARE_API_URL environment variable set; keeping existing js/env.js");
}
