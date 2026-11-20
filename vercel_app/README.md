# CyberSLM Vercel surface

This is a separate Next.js access point; it does not replace the local Streamlit app. Deterministic
CSV/JSON/JSONL/Parquet parsing and evidence counts run in the Vercel Node runtime. Model requests are
proxied only when both `CYBERSLM_BACKEND_URL` (authenticated HTTPS) and
`CYBERSLM_BACKEND_TOKEN` are configured as server-only Vercel environment variables.

Without that remote backend, the deployed site remains honest: evidence verification works, while
Qwen and CyberSLM-35M are shown as unavailable. Localhost Ollama/LM Studio cannot be reached by
Vercel. The in-process rate limiter is best-effort per function instance; production use should add
a platform firewall rule or shared rate-limit store.

```powershell
cd vercel_app
npm ci
npm run lint
npm run typecheck
npm test
npm run build
npm run dev
```
