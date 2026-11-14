# Security and data handling

- Raw/processed datasets, split exports, checkpoints, logs, secrets, and experiment artifacts are ignored by Git.
- Only the explicitly fictional fixture may be tracked under `data/fixtures`.
- Do not execute macros, scripts, binaries, links, or embedded objects from reports.
- Scan and hash incoming files, record acquisition source and license, and process them in an isolated environment.
- Do not commit professor-provided, proprietary, sensitive, or personally identifiable material without written authorization.
- Do not assume an index repository's license covers third-party linked reports; verify each original source.
- Review training text for credentials, personal data, malicious payloads, licensing restrictions, and benchmark leakage.
- Store secrets only in local `.env` or an approved secret manager. `.env.example` contains names, not values.

Before real-data integration, document the data controller, access rules, retention/deletion process, incident contact, approved licenses, and whether redistribution or model training is permitted.

