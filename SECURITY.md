# Security

## Supported versions

This is a research prototype. Only the latest version is supported, and fixes
are not backported.

## The demo server is a local tool

`adaptive_thinking/server.py` and `adaptive_thinking/static/index.html` are a
single-user demo. Their design assumes they are never reachable from anyone
else's machine:

- **It binds `127.0.0.1` only** (the port is `PORT` in `.env`). Do not change
  it to listen on other interfaces, or put it behind a reverse proxy, a tunnel
  or a port forward.
- **There is no authentication.**
  - Anyone who can reach the port can run turns on your model endpoint, at
    your cost.
  - They can continue or reset any conversation whose session id they know.
- **It is single-user.**
  - Conversations are kept in memory, keyed by a session id the client
    chooses, and nothing bounds their number.
  - The `inline` query parameter on `/chat` changes a **process-wide**
    setting (`KEEP_BLOCKS_INLINE`) for every session.
- **It logs what you type.** Every question and a selection of engine events
  are appended to `events.jsonl` at the repository root:
  - the answer text,
  - the gate decisions,
  - the thinking monologues,
  - the text of any endpoint error recorded in a turn's stats.

  That file is gitignored. Treat it as containing everything you asked.
- **The UI renders some model output as HTML.** Most text in
  `adaptive_thinking/static/index.html` goes through `textContent`. The gate
  summary line drawn for each scored window is the exception: it inserts the
  model's candidate token through `innerHTML`, and `JSON.stringify` does not
  escape `<` or `>`. A crafted or unlucky output could therefore inject markup
  or script into the page. This is acceptable only because the page is served
  to its own user on localhost.

## Secrets

- The endpoint URL and API key live only in `.env`, which is gitignored.
  `.env.example` holds placeholders. Never commit a real `.env`.
- The API key travels in an `Authorization` header and is never written to
  logs.
- The endpoint's **URL** can appear in error text. An HTTP error from the
  endpoint includes the request URL, and that text can reach the `degraded`
  field of a turn's stats, `events.jsonl`, the UI and the zebra result files
  (`experiments/zebra/zebra_results.jsonl`, which is not tracked). Check logs
  and result files before sharing them.
- `python -m adaptive_thinking.llm` prints the configured endpoint URL as part
  of its smoke test.
- The offline tests never read `.env` and make no network requests.

## Reporting a vulnerability

Please report security issues privately. Use GitHub's private vulnerability
reporting: the **Security** tab of this repository, then **Report a
vulnerability**. It opens a draft security advisory visible only to the
maintainers. Please do not open a public issue for a security problem.

Include:

- what is affected,
- how to reproduce it,
- what an attacker could do with it.
