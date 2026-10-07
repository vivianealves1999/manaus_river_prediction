# Data sources

Everything here is public and requires no credentials. Where a source is awkward to reach,
the reason and the workaround are recorded so the next person does not repeat the search.

## 1. Target: Rio Negro stage at Manaus

### 1.1 ANA / HidroWeb — station 14990000 (Porto de Manaus)

The authoritative Brazilian national record, operated by the Agência Nacional de Águas e
Saneamento Básico and the Serviço Geológico do Brasil.

```
http://telemetriaws1.ana.gov.br/ServiceANA.asmx/HidroSerieHistorica
    ?codEstacao=14990000
    &dataInicio=01/01/1950&dataFim=31/12/1954
    &tipoDados=1            # 1 = Cotas (stage), 2 = rainfall, 3 = discharge
    &nivelConsistencia=     # blank = both raw (1) and consisted (2)
```

Returns XML with one `SerieHistorica` record per (station, month, consistency level); each
record carries up to 31 daily readings in `Cota01`…`Cota31`, **in centimetres**.

**Coverage obtained: 1902-09-15 → 2014-12-30, 41,009 daily observations.** That is 112 years
— comfortably beyond the "since 1950" the project asked for.

Two traps:

1. **Long date ranges return an empty result set, not an error.** A request for 1950–2026
   yields nothing; 1950–1954 yields the data. Hence the 5-year chunking in `ana_client.py`.
2. **Each month appears twice**, at `NivelConsistencia` 1 (raw) and 2 (quality-checked).
   Keep 2 where present, fall back to 1.

### 1.2 Why a second source was needed

The legacy service stops delivering Manaus data at **2014-12**. It still serves upstream
stations past that date, so this is specific to this station, not a global cutoff. Every
other ANA route to recent data is now gated:

| Route | Result |
|---|---|
| `HidroSerieHistorica` with `nivelConsistencia=1`, 2015–2026 | empty |
| Alternate station codes (14990010, 14991000, 14980000, 14996000) | empty |
| `DadosHidrometeorologicos` (telemetry, 15-min) | works, but only retains ~the last month |
| `www.ana.gov.br/hidrowebservice/...` (new REST API) | **401** — requires registration token |
| `snirh.gov.br/hidroweb/rest/api/documento/convencionais` | **401** — requires API token |
| `apps.sgb.gov.br/hidro-series/` | R/Shiny app, websocket session — not scriptable |

### 1.3 portodemanaus.com.br — daily archive, 2000–present

The port authority publishes the official Porto de Manaus ruler reading daily. **Same
physical gauge as ANA 14990000.**

The site is WordPress + JetEngine; each month is a `nivel-do-rio` custom post carrying a
daily table, surfaced through a JetSmartFilters listing that accepts filters in the URL:

```
https://portodemanaus.com.br/nivel-do-rio-negro/
    ?jsf=jet-engine:result
    &tax=ano:<year_term_id>;meses:<month_term_id>
```

Taxonomy term ids are discovered at run time from the public WP REST API
(`/wp-json/wp/v2/ano` and `/wp-json/wp/v2/meses`) — they are not sequential, so hard-coding
them would break.

**Coverage obtained: 2000-01-01 → present, ~9,700 daily observations.**

Three quirks, all handled in `porto_manaus.py`:

- **Not unit-consistent.** Recent years publish metres (`24,09`); older years publish whole
  centimetres (`1641`). Since the gauge can never read below 5 m or above 35 m, magnitude
  disambiguates them unambiguously.
- **Transcription typos.** A handful of readings sit metres away from both neighbours (e.g.
  2012-12-14 is 7.48 m off). The Rio Negro is a heavily damped system — day-to-day change is
  centimetres — so an isolated spike beyond 0.6 m/day is a digit error, not hydrology.
- **Month headings are not reliably rendered** in the multi-month paged view, which silently
  mis-assigns whole months. Fetching one month per request avoids the problem entirely.

### 1.4 Splicing the two

Validated over the **5,459-day overlap** (2000–2014):

| Statistic | Value |
|---|---|
| Correlation | **0.99946** |
| Mean difference (Porto − ANA) | **−0.0024 m** |
| Std. dev. of difference | **0.127 m** |
| Days differing by > 0.9 m | 4 (all Porto-side typos, now filtered) |

They are the same gauge. `build_target_daily()` takes ANA wherever it exists and appends
Porto de Manaus only for dates ANA lacks.

## 2. Upstream and neighbouring gauges

> The project brief asked whether other Amazon rivers influence the Rio Negro, and where to
> get their data. They do, strongly, and this is the answer to both.

**The Rio Negro at Manaus is not a free-draining river.** It joins the Rio Solimões a few
kilometres downstream at the Meeting of the Waters. The Solimões carries several times the
Negro's discharge, and when it runs high it acts as a **hydraulic dam**: it backs water up
into the Negro and raises the stage at Manaus independently of anything happening in the
Negro's own catchment. A large part of what the Manaus gauge measures is therefore *the
Solimões*, not the Negro.

This makes the Solimões gauge at Manacapuru the single strongest non-local predictor
available, and it is why the panel carries neighbouring rivers at all.

All are fetched through the same ANA endpoint, just with a different `codEstacao`:

| Code | Station | River | Why it is included |
|---|---|---|---|
| `14100000` | Manacapuru | Solimões | ~80 km upstream of the confluence. The backwater effect described above — the strongest external predictor. |
| `14710000` | Caracaraí | Branco | Largest tributary of the Rio Negro. Northern-hemisphere rainfall regime, so it peaks out of phase with the southern basin. |
| `13150000` | Tabatinga | Solimões | Far-upstream Solimões at the Peru/Colombia border. ~2–3 month travel time to Manaus, which is a useful lead at exactly this project's horizon. |
| `15400000` | Porto Velho | Madeira | Southern Amazon regime; proxy for the southern-basin monsoon. Weakest of the four for Manaus but cheap to carry. |

Note the lag rule from `features.py` still applies: these enter the design matrix shifted by
at least the forecast horizon, because their own future values are unknown at forecast time.

### Other gauges worth adding

The same endpoint serves the whole national network. Useful additions:

- `14840000` Itacoatiara (Amazonas, downstream) — for validating routing
- `14526000` Moura / `14420000` Serrinha (Rio Negro upstream) — pure Negro signal
- `15630000` Manicoré (Madeira)

Station codes can be browsed at [snirh.gov.br/hidroweb](https://www.snirh.gov.br/hidroweb/).

## 3. Climate indices — NOAA PSL

Fixed-width ASCII, one row per year with 12 monthly columns:

| Index | URL | Coverage | What it measures |
|---|---|---|---|
| **ONI** | `psl.noaa.gov/data/correlation/oni.data` | 1950– | Niño 3.4 SST anomaly — **the ENSO index** |
| **TNA** | `.../tna.data` | 1948– | Tropical North Atlantic SST anomaly |
| **TSA** | `.../tsa.data` | 1948– | Tropical South Atlantic SST anomaly |
| **SOI** | `.../soi.data` | 1951– | Southern Oscillation Index (atmospheric ENSO) |
| **AMO** | `.../amon.us.data` | 1948–2023 | Atlantic Multidecadal Oscillation |

The format has a trap: a sentinel line after the data block declares the missing value
(commonly `-99.99`). `parse_psl_ascii()` reads that sentinel rather than assuming it, and
applies a magnitude guard as a backstop.

**Why TNA and TSA are here and not just ONI.** The Amazon's two most severe recent droughts
before 2023 — 2005 and 2010 — occurred with ENSO near neutral. They were driven by a warm
Tropical North Atlantic displacing the ITCZ northward. A model that watches only the Pacific
would have missed both. See `docs/MODELING.md` for how much each actually contributes.

## 4. Sources considered and rejected

| Source | Why not |
|---|---|
| ANA HidroWebService REST (new) | 401 without a registration token |
| SNIRH document/ZIP endpoint | 401 without an API token |
| SGB `apps.sgb.gov.br/hidro-series` | R/Shiny websocket app; no stable HTTP API |
| SO-HYBAM (IRD) | Excellent Amazon-basin series, but registration required |
| GRDC | Discharge not stage, registration required, long lag |
| SGB/CPRM SACE bulletins | PDF only; good for cross-checking headline numbers, not for a pipeline |

## 5. Refreshing

```bash
python main.py --refresh-data     # re-download everything (~6 min, mostly Porto de Manaus)
python main.py --data-only        # rebuild the panel without modelling
```

All raw downloads are cached under `data/raw/` and `data/external/`, so ordinary re-runs are
offline. Requests to portodemanaus.com.br are rate-limited to one per 0.7 s.
