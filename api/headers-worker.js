/* Security-headers Worker for quantenergy.tech — Cloudflare deployment recipe.
 *
 * WHY THIS EXISTS
 *   The site is served by GitHub Pages, which does not support custom HTTP
 *   response headers. Four recommended headers are therefore missing today:
 *     - Content-Security-Policy          (the site already ships a <meta> CSP
 *                                         on every page as a stopgap; a real
 *                                         header is authoritative and supports
 *                                         frame-ancestors / report-to)
 *     - Strict-Transport-Security        (cannot be done via <meta> at all)
 *     - X-Content-Type-Options           (same)
 *     - Permissions-Policy               (same)
 *   This Worker adds them by proxying: Cloudflare edge -> GitHub Pages origin.
 *
 * DEPLOYMENT (one-time, ~10 minutes, free tier is enough)
 *   1. In the Cloudflare dashboard, add the domain quantenergy.tech as a zone
 *      (needs the domain's nameservers to point to Cloudflare).
 *   2. In the zone's DNS settings, keep the existing CNAME/A records that point
 *      at GitHub Pages (<user>.github.io), but switch the proxy status to
 *      "Proxied" (orange cloud).
 *   3. Create this Worker (Workers & Pages -> Create Worker -> paste this file),
 *      name it e.g. "quantenergy-headers".
 *   4. Add a route in the zone: quantenergy.tech/* -> quantenergy-headers.
 *   5. Verify:  curl -sI https://quantenergy.tech/ | grep -iE "content-security|strict-transport|x-content-type|permissions-policy"
 *
 * NOTES
 *   - GitHub Pages already forces HTTPS and redirects HTTP, but HSTS removes
 *     even that first redirect round-trip and locks the policy in the browser.
 *   - The CSP below mirrors the <meta> CSP shipped on every page. Once this
 *     Worker is live you may keep the meta tags (harmless, belt-and-braces) or
 *     remove them; the header wins when both are present.
 *   - If you later move hosting off GitHub Pages, configure these same headers
 *     at the new host and retire this Worker.
 *
 * This file is a recipe, not a deployed artifact of the site; it lives in the
 * repo so the exact policy is versioned next to the code it protects.
 */

const GITHUB_PAGES_ORIGIN = "hongping-zh.github.io"; // the CNAME target

const HEADERS = {
  "content-security-policy":
    "default-src 'self'; " +
    "script-src 'self' 'unsafe-inline'; " +
    "style-src 'self' 'unsafe-inline'; " +
    "img-src 'self' data: https:; " +
    "font-src 'self' data:; " +
    "connect-src 'self' https://ecocompute-estimator.zhanghongping1982.workers.dev; " +
    "base-uri 'self'; form-action 'self'; object-src 'none'; frame-src 'none'; " +
    "frame-ancestors 'none'", // header-only directive, ignored by the meta tag
  "strict-transport-security": "max-age=31536000; includeSubDomains; preload",
  "x-content-type-options": "nosniff",
  "permissions-policy":
    "camera=(), microphone=(), geolocation=(), payment=(), usb=(), " +
    "interest-cohort=()",
  "referrer-policy": "strict-origin-when-cross-origin",
};

export default {
  async fetch(request) {
    const url = new URL(request.url);

    // Only proxy the production host; anything else passes through untouched.
    if (url.hostname !== "quantenergy.tech") return fetch(request);

    // Rewrite the origin to GitHub Pages, preserving path and query.
    const originUrl = new URL(request.url);
    originUrl.hostname = GITHUB_PAGES_ORIGIN;

    const upstream = await fetch(new Request(originUrl, request));
    const response = new Response(upstream.body, upstream);
    for (const [k, v] of Object.entries(HEADERS)) response.headers.set(k, v);
    return response;
  },
};
