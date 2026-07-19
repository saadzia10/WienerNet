#!/usr/bin/env python
"""Assemble the manuscript-grade HTML report with the figure embedded as a
data URI. Read-only w.r.t. the codebase."""
import base64, os
HERE = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(HERE, "fig_error_structure.png"), "rb") as f:
    b64 = base64.b64encode(f.read()).decode()
IMG = f"data:image/png;base64,{b64}"

HTML = r"""<title>Nighttime NEE residuals: heteroscedastic & non-Gaussian</title>
<style>
:root{
  --bg:#fbfcfd; --surface:#ffffff; --ink:#16191e; --muted:#57616f;
  --faint:#e4e9ef; --rule:#c9d2db; --accent:#0f6fb0; --accent2:#c8551f;
  --serif:"Palatino Linotype","Book Antiqua",Palatino,Georgia,"Times New Roman",serif;
  --mono:ui-monospace,"SF Mono",Menlo,Consolas,"Liberation Mono",monospace;
}
@media (prefers-color-scheme:dark){
  :root{--bg:#0e1217;--surface:#151b22;--ink:#e7ebf0;
    --muted:#9aa7b5;--faint:#243039;--rule:#33414d;--accent:#57aee2;--accent2:#e58150;}
}
:root[data-theme="light"]{--bg:#fbfcfd;--surface:#ffffff;--ink:#16191e;--muted:#57616f;
  --faint:#e4e9ef;--rule:#c9d2db;--accent:#0f6fb0;--accent2:#c8551f;}
:root[data-theme="dark"]{--bg:#0e1217;--surface:#151b22;--ink:#e7ebf0;--muted:#9aa7b5;
  --faint:#243039;--rule:#33414d;--accent:#57aee2;--accent2:#e58150;}

*{box-sizing:border-box}
body{background:var(--bg);color:var(--ink);font-family:var(--serif);
  line-height:1.62;margin:0;-webkit-font-smoothing:antialiased;}
.wrap{max-width:820px;margin:0 auto;padding:64px 28px 96px;}
.eyebrow{font-family:var(--mono);font-size:11.5px;letter-spacing:.14em;
  text-transform:uppercase;color:var(--accent);margin:0 0 18px;}
h1{font-size:33px;line-height:1.18;font-weight:600;margin:0 0 14px;
  text-wrap:balance;letter-spacing:-.01em;}
.lede{font-size:18px;color:var(--muted);margin:0 0 22px;text-wrap:balance;}
.meta{font-family:var(--mono);font-size:12px;color:var(--muted);
  border-top:1px solid var(--faint);border-bottom:1px solid var(--faint);
  padding:11px 0;display:flex;flex-wrap:wrap;gap:6px 20px;margin:0 0 40px;}
.meta b{color:var(--ink);font-weight:600;}
.abstract{background:var(--surface);border:1px solid var(--faint);
  border-left:3px solid var(--accent);border-radius:3px;padding:18px 22px;
  font-size:15.5px;margin:0 0 44px;}
.abstract .lbl{font-family:var(--mono);font-size:10.5px;letter-spacing:.13em;
  text-transform:uppercase;color:var(--accent);display:block;margin-bottom:7px;}
h2{font-size:13px;font-family:var(--mono);letter-spacing:.06em;text-transform:uppercase;
  color:var(--ink);margin:52px 0 4px;padding-bottom:8px;border-bottom:1px solid var(--rule);}
h2 .n{color:var(--accent);margin-right:12px;}
h3{font-size:17px;font-weight:600;margin:30px 0 2px;color:var(--ink);}
p{margin:14px 0;}
em{font-style:italic;}
strong{font-weight:600;}
.eq{font-family:var(--serif);font-style:italic;font-size:16px;text-align:center;
  background:var(--surface);border:1px solid var(--faint);border-radius:3px;
  padding:14px 10px;margin:18px 0;overflow-x:auto;}
.eq .sub{font-size:.72em;font-style:normal;vertical-align:sub;}
figure{margin:34px 0 8px;}
figure img{width:100%;height:auto;display:block;border:1px solid var(--faint);
  border-radius:4px;background:#fff;}
figcaption{font-size:13.5px;color:var(--muted);margin-top:12px;line-height:1.5;}
figcaption b{color:var(--ink);font-weight:600;}
.tbl-scroll{overflow-x:auto;margin:20px 0;}
table{border-collapse:collapse;width:100%;font-size:13.5px;
  font-variant-numeric:tabular-nums;}
caption{caption-side:top;text-align:left;font-family:var(--mono);font-size:11px;
  letter-spacing:.06em;text-transform:uppercase;color:var(--muted);
  margin-bottom:9px;}
thead th{border-top:1.6px solid var(--ink);border-bottom:1px solid var(--ink);
  text-align:right;padding:7px 12px;font-weight:600;font-family:var(--mono);
  font-size:11.5px;letter-spacing:.02em;}
thead th:first-child{text-align:left;}
tbody td{border-bottom:1px solid var(--faint);padding:6px 12px;text-align:right;}
tbody td:first-child{text-align:left;font-family:var(--mono);font-size:12px;}
tbody tr:last-child td{border-bottom:1.6px solid var(--ink);}
tr.pooled td{font-weight:700;background:color-mix(in srgb,var(--accent) 7%,transparent);}
.hi{color:var(--accent2);font-weight:600;}
.callout{background:var(--surface);border:1px solid var(--faint);
  border-left:3px solid var(--accent2);border-radius:3px;padding:16px 20px;
  font-size:14.5px;margin:30px 0;}
.callout .lbl{font-family:var(--mono);font-size:10.5px;letter-spacing:.13em;
  text-transform:uppercase;color:var(--accent2);display:block;margin-bottom:6px;}
ul{margin:14px 0;padding-left:22px;}
li{margin:8px 0;}
.foot{margin-top:56px;padding-top:16px;border-top:1px solid var(--faint);
  font-family:var(--mono);font-size:11.5px;color:var(--muted);line-height:1.7;}
a{color:var(--accent);}
</style>

<div class="wrap">
  <p class="eyebrow">WienerNet · Flux error structure</p>
  <h1>Nighttime NEE residuals are heteroscedastic and non-Gaussian</h1>
  <p class="lede">The physics-model residual scales linearly with flux magnitude and
     follows a double-exponential (Laplace) law — not the Gaussian assumed by a
     constant-variance loss.</p>
  <div class="meta">
    <span><b>Sites</b> 6 UK fen / peatland towers</span>
    <span><b>n</b> 92,063 nighttime half-hours</span>
    <span><b>Model</b> Reco = Lloyd&ndash;Taylor(T<span class="sub">a</span>; E&#8320;, r&#8342;)</span>
  </div>

  <div class="abstract">
    <span class="lbl">Abstract</span>
    We characterise the residual between observed nighttime NEE and the analytic
    Lloyd&ndash;Taylor respiration used as the drift of the WienerNet SDE. Across six
    towers the residual is strongly <strong>heteroscedastic</strong> &mdash; its scale
    grows linearly with predicted flux (SD&nbsp;&asymp;&nbsp;0.24&nbsp;+&nbsp;0.30&thinsp;&middot;&thinsp;flux,
    <em>R</em><sup>2</sup>=0.99) &mdash; and its shape is <strong>leptokurtic and
    double-exponential</strong>: after the flux-dependence is removed the standardised
    residual still has excess kurtosis 18.7 and 3&sigma; tail probability 0.014, matching
    a Laplace (0.014) rather than a Gaussian (0.003). Normality is rejected at
    <em>p</em>&nbsp;&lt;&nbsp;10<sup>&minus;50</sup>. Under temporal aggregation the random
    component cancels, and relative uncertainty of the summed flux falls from ~42% at
    the half-hour to ~6% monthly. These properties reproduce the eddy-covariance error
    model of Hollinger &amp; Richardson and argue for a heteroscedastic, heavy-tailed
    likelihood in the SDE noise term.
  </div>

  <h2><span class="n">1</span>Residual definition</h2>
  <p>At night GPP&nbsp;=&nbsp;0, so the physics prediction is ecosystem respiration under the
    Reichstein form of Lloyd&ndash;Taylor, with per-window parameters (E&#8320;,&nbsp;r&#8342;)
    fitted offline via REddyProc:</p>
  <div class="eq">Reco(T<span class="sub">a</span>) = r&#8342; &middot; exp&#8202;[&#8202;E&#8320;
    (1/(T<span class="sub">ref</span>+T&#8320;) &minus; 1/(T<span class="sub">a</span>+T&#8320;))&#8202;],
    &nbsp;&nbsp; T<span class="sub">ref</span>=10&deg;C, T&#8320;=46.02&deg;C</div>
  <div class="eq">residual &nbsp;=&nbsp; NEE<span class="sub">obs</span> &minus; Reco(T<span class="sub">a</span>)</div>
  <p>This residual is precisely the quantity Hollinger &amp; Richardson used to
    characterise flux uncertainty; it conflates random measurement error with model
    structural error, so we separate the systematic (mean) part from the random
    (scatter) part where it matters.</p>

  <figure>
    <img alt="Four-panel figure: residual density vs Gaussian and Laplace, tail probabilities, heteroscedastic scale versus flux, and relative uncertainty versus aggregation window." src="__IMG__">
    <figcaption><b>Figure 1.</b> Error structure of the pooled physics residual
      (<em>n</em>&nbsp;=&nbsp;92,063). <b>(A)</b> Density of the flux-standardised residual on a
      log axis; the Laplace fit captures the tent-shaped peak and heavy tails the
      Gaussian misses. <b>(B)</b> Tail exceedance <em>P</em>(|z|&gt;k): the empirical curve
      tracks the Laplace prediction and departs sharply from the Gaussian beyond 2&sigma;.
      <b>(C)</b> Local residual scale rises linearly with predicted flux
      (heteroscedasticity; <em>R</em><sup>2</sup>=0.99). <b>(D)</b> Relative uncertainty of the
      summed flux falls under aggregation but above the ideal 1/&radic;N line, because the
      half-hourly residual is mildly autocorrelated (lag-1 = 0.29).</figcaption>
  </figure>

  <h2><span class="n">2</span>Results</h2>

  <h3>2.1&ensp;The residual is heteroscedastic (scale &prop; flux)</h3>
  <p>Binning by predicted flux, the residual standard deviation increases <em>linearly</em>
    from ~0.65 at low respiration to ~3.6 at the highest bin &mdash; a 5.5&times; range. A
    single line, SD&nbsp;&asymp;&nbsp;0.24&nbsp;+&nbsp;0.30&thinsp;&middot;&thinsp;flux, explains
    99% of the between-bin variance (Fig.&nbsp;1C). The slope is positive and significant at
    every site, and |residual| correlates with flux at <em>r</em>&nbsp;=&nbsp;0.42
    (<em>p</em>&nbsp;&approx;&nbsp;0).</p>
  <div class="tbl-scroll">
  <table>
    <caption>Table 1 &mdash; Heteroscedasticity by site: slope of SD(residual) on flux</caption>
    <thead><tr><th>Site</th><th>n</th><th>slope&nbsp;dSD/dflux</th><th>R&sup2;</th><th>corr(|r|,&nbsp;flux)</th></tr></thead>
    <tbody>
      <tr><td>rosedene</td><td>25,335</td><td>0.380</td><td>0.956</td><td>0.435</td></tr>
      <tr><td>redmere_2</td><td>21,436</td><td>0.303</td><td>0.964</td><td>0.478</td></tr>
      <tr><td>woodwalton</td><td>22,250</td><td>0.194</td><td>0.792</td><td>0.302</td></tr>
      <tr><td>great_fen</td><td>9,918</td><td>0.239</td><td>0.763</td><td>0.272</td></tr>
      <tr><td>wicken_fen</td><td>7,660</td><td>0.251</td><td>0.875</td><td>0.411</td></tr>
      <tr><td>redmere_1</td><td>5,464</td><td>0.289</td><td>0.958</td><td>0.461</td></tr>
      <tr class="pooled"><td>pooled</td><td>92,063</td><td>0.303</td><td>0.986</td><td>0.420</td></tr>
    </tbody>
  </table>
  </div>

  <h3>2.2&ensp;The shape is double-exponential, not Gaussian</h3>
  <p>The raw residual is extremely leptokurtic (excess kurtosis 29.9). Crucially, this is
    <em>not</em> merely an artefact of mixing flux magnitudes: after dividing each residual
    by its local (flux-bin) scale, the standardised residual still has excess kurtosis 18.7
    and a 3&sigma; tail probability of 0.014 &mdash; essentially the Laplace value (0.014) and
    5&times; the Gaussian (0.003). A Laplace model beats a Gaussian by
    <span class="hi">&Delta;AIC &gt; 28,000</span>, and Shapiro&ndash;Wilk, Anderson&ndash;Darling
    and Jarque&ndash;Bera all reject normality at <em>p</em>&nbsp;&lt;&nbsp;10<sup>&minus;50</sup>.
    The distribution is also right-skewed (positive spikes from storage/advection), so the
    true law is heavy-tailed <em>and</em> asymmetric.</p>
  <div class="tbl-scroll">
  <table>
    <caption>Table 2 &mdash; Distribution shape: Gaussian vs Laplace</caption>
    <thead><tr><th>Residual set</th><th>excess&nbsp;kurtosis</th><th>&Delta;AIC (N&minus;L)</th>
      <th>P(|z|&gt;3)&nbsp;emp.</th><th>Gaussian</th><th>Laplace</th></tr></thead>
    <tbody>
      <tr><td>raw</td><td>29.9</td><td>+47,450</td><td>0.0187</td><td>0.0027</td><td>0.0144</td></tr>
      <tr><td>winsorised&nbsp;0.5%</td><td>5.3</td><td>+32,622</td><td>0.0244</td><td>0.0027</td><td>0.0144</td></tr>
      <tr class="pooled"><td>flux-standardised</td><td>18.7</td><td>+28,229</td><td>0.0143</td><td>0.0027</td><td>0.0144</td></tr>
    </tbody>
  </table>
  </div>
  <p style="font-size:12.5px;color:var(--muted);margin-top:-6px;">
    &Delta;AIC&nbsp;&gt;&nbsp;0 favours Laplace. The flux-standardised row isolates distributional
    <em>shape</em> from heteroscedasticity; its empirical 3&sigma; tail matches Laplace to three decimals.</p>

  <h3>2.3&ensp;Random error cancels under aggregation</h3>
  <p>Summing the residual over increasing windows, the relative uncertainty of the flux
    sum falls from <span class="hi">~42% at the half-hour to ~9% weekly and ~6% monthly</span>
    (Table&nbsp;3, Fig.&nbsp;1D) &mdash; the canonical eddy-covariance result that half-hourly noise
    largely cancels in longer sums. The decline is <em>slower</em> than the ideal 1/&radic;N,
    however: the half-hourly residual carries lag-1 autocorrelation 0.29, so errors are not
    fully independent, and the flux-dependent systematic bias (Table&nbsp;1 context) does not
    cancel at all.</p>
  <div class="tbl-scroll">
  <table>
    <caption>Table 3 &mdash; Relative uncertainty of the summed flux vs aggregation window</caption>
    <thead><tr><th>Window</th><th>half-hours (N)</th><th>relative uncertainty</th></tr></thead>
    <tbody>
      <tr><td>30 min (native)</td><td>1</td><td>42.1%</td></tr>
      <tr><td>~1 night</td><td>10</td><td>22.6%</td></tr>
      <tr><td>~1 day</td><td>48</td><td>15.2%</td></tr>
      <tr><td>~1 week</td><td>336</td><td>9.3%</td></tr>
      <tr><td>~1 month</td><td>1,440</td><td>6.1%</td></tr>
      <tr class="pooled"><td>~1 year-equiv.</td><td>17,520</td><td>3.2%</td></tr>
    </tbody>
  </table>
  </div>

  <h2><span class="n">3</span>Why the error takes this form</h2>
  <p><strong>Heteroscedasticity</strong> follows from the physics of turbulent sampling: the
    <em>relative</em> (multiplicative) flux error is approximately constant, so the
    <em>absolute</em> error standard deviation grows in proportion to the flux magnitude.
    A larger respiration signal is measured over the same averaging window with the same
    fractional turbulent sampling error, giving a larger absolute scatter &mdash; exactly the
    linear SD&nbsp;&prop;&nbsp;flux of Fig.&nbsp;1C.</p>
  <p><strong>Non-Gaussianity</strong> follows from that same variability. The instantaneous
    error is approximately Gaussian, but its <em>variance changes</em> with turbulence,
    footprint and meteorological state. A mixture of Gaussians with varying variance
    (a scale mixture) is necessarily leptokurtic &mdash; heavier-tailed than any single
    Gaussian &mdash; and converges toward the double-exponential. This is why the Laplace
    shape survives even after we standardise away the flux-dependence: the heavy tail is
    intrinsic to a varying-variance sampling process, not a binning artefact.</p>

  <div class="callout">
    <span class="lbl">Implications for the SDE noise term</span>
    A constant-variance Gaussian likelihood is doubly misspecified here. (i) The noise
    scale must be <strong>input-dependent</strong> &mdash; the per-sample &sigma;-head should
    reproduce the 0.30 flux-slope; a homoscedastic &sigma; is rejected at <em>R</em><sup>2</sup>=0.99.
    (ii) The likelihood should be <strong>heavy-tailed</strong> &mdash; a Student-t or Laplace
    (L1) NLL matches the tails a Gaussian under-weights 5-fold; the observed right-skew
    further motivates an asymmetric form or a distribution-matching (MMD) term.
    (iii) Half-hourly R&sup2;/RMSE are floored by ~40% irreducible measurement noise, so
    <strong>weekly&ndash;monthly aggregated sums</strong> (relative uncertainty ~6&ndash;9%) are the
    honest evaluation scale.
  </div>

  <div class="foot">
    Reference &mdash; Hollinger &amp; Richardson (2005) <em>Tree Physiology</em>;
    Richardson et&nbsp;al. (2006) <em>Agric. For. Meteorol.</em><br>
    Exploratory analysis over final_night_data parquet (6 sites); residual =
    NEE<sub>obs</sub> &minus; Lloyd&ndash;Taylor(T<sub>a</sub>; E&#8320;, r&#8342;). Pooled n = 92,063.
  </div>
</div>
"""

out = os.path.join(HERE, "report.html")
with open(out, "w") as f:
    f.write(HTML.replace("__IMG__", IMG))
print("wrote", out, "bytes:", os.path.getsize(out))
