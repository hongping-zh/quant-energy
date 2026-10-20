/* =============================================================================
 * EcoCompute — data source adapter (the seam between the two layers)
 *
 * PUBLIC RESEARCH LAYER (now):  window.ECO_API_BASE is unset, so every read
 *   resolves to a static JSON file served by GitHub Pages. The site works
 *   fully offline and needs no backend.
 *
 * PRODUCT / SERVICE LAYER (later): deploy the API, then inject
 *   <script>window.ECO_API_BASE = "https://api.example/v1";</script>
 *   before this file. The same pages then read from the API. No page code
 *   changes, no rewrites — only this adapter switches.
 *
 * The two layers are coupled ONLY through the JSON contracts:
 *   - energy.schema.json   (one measurement report)
 *   - replications.json    (public replication entries)
 * The service layer MUST emit the same schemas, otherwise its results cannot
 * be compared against the published curves.
 * ========================================================================== */
(function () {
  "use strict";

  var BASE = (typeof window !== "undefined" && window.ECO_API_BASE) || "";

  var PATHS = {
    replications: "/replications/replications.json"
  };

  function url(key) {
    var p = PATHS[key];
    if (!p) throw new Error("EcoData: unknown key '" + key + "'");
    return BASE ? BASE + p : p;
  }

  /**
   * Read one contract payload.
   * Returns a Promise; in static mode reads the file, in API mode GETs it.
   * Callers keep their own fallback content — the public layer must show real
   * data even when the network or the API is unavailable.
   */
  function fetchJson(key) {
    var u = url(key);
    return fetch(u, { cache: "no-store" }).then(function (r) {
      if (!r.ok) throw new Error("EcoData " + key + ": HTTP " + r.status);
      return r.json();
    }).catch(function (err) {
      console.warn("[EcoData] fetch failed for", key + ":", err.message || err);
      throw err;
    });
  }

  window.EcoData = {
    base: BASE,
    mode: BASE ? "api" : "static",
    paths: PATHS,
    url: url,
    fetchJson: fetchJson
  };
})();
