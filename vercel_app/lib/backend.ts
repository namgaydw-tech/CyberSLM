const MAX_RESPONSE_BYTES = 1_000_000;

export function backendConfigured() {
  const url = process.env.CYBERSLM_BACKEND_URL;
  const token = process.env.CYBERSLM_BACKEND_TOKEN;
  return Boolean(url && token && /^https:\/\//.test(url));
}

export async function callBackend(path: string, body: unknown) {
  const baseUrl = process.env.CYBERSLM_BACKEND_URL;
  const token = process.env.CYBERSLM_BACKEND_TOKEN;
  if (!baseUrl || !token || !/^https:\/\//.test(baseUrl)) {
    throw new Error("Remote model backend is not configured with an authenticated HTTPS URL.");
  }
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), 45_000);
  try {
    const response = await fetch(`${baseUrl.replace(/\/$/, "")}${path}`, {
      method: "POST",
      headers: {
        authorization: `Bearer ${token}`,
        "content-type": "application/json",
      },
      body: JSON.stringify(body),
      signal: controller.signal,
      cache: "no-store",
    });
    const text = await response.text();
    if (text.length > MAX_RESPONSE_BYTES) throw new Error("Backend response exceeded 1 MB.");
    if (!response.ok) throw new Error(`Backend returned HTTP ${response.status}.`);
    return JSON.parse(text) as Record<string, unknown>;
  } finally {
    clearTimeout(timer);
  }
}
