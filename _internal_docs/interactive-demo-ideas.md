# Interactive 3D/2D Demo Ideas for Blog Posts

Reference implementation: `posts/seasons/index.qmd` — Three.js orbital mechanics demo with smooth animation, parameter controls, camera modes, dynamic geometry, and CSS starfield background. The full version at `/sunearthmoon/seasons.html` adds OrbitControls, speed sliders, pause/play, and collapsible panels.

---

## Priority 1: High Impact, Inherently Spatial

### 1. Gradient Descent on a Loss Surface

**Post:** `gradientdescent`
**Priority:** Highest — universal ML concept, everyone benefits

**Description:** A glowing ball rolls downhill on a 3D loss surface (Rosenbrock, saddle point, or quadratic bowl). The ball leaves a fading trail showing its path. With high learning rate it overshoots and oscillates; with low LR it creeps slowly; with momentum it accelerates through flat regions. The surface itself is translucent so you can see the ball from any angle.

**Axes:** X = parameter 1 (e.g., weight), Y = parameter 2 (e.g., bias), Z (height) = loss value

**2D alternative:** Top-down contour plot with animated dot tracing the gradient path. Contour lines + arrow trail. Simpler but loses the visceral "rolling downhill" intuition. Could use D3.js or plain Canvas.

**Preferred tech:** Three.js — the 3D surface is the whole point; camera orbit lets you see the landscape from different angles (bird's eye for contours, side view for steepness).

**Implementation sketch:**
1. Generate surface mesh from an analytic function (e.g., `z = (1-x)^2 + 100*(y-x^2)^2` for Rosenbrock) using `PlaneGeometry` with vertex displacement
2. Color the mesh by height using a sequential colormap (vertex colors or a gradient texture)
3. Place a `SphereGeometry` ball at a starting point; each animation frame compute gradient analytically, update position: `x -= lr * dz/dx`
4. Store trail positions in a `BufferGeometry` line that fades via vertex alpha
5. Controls: learning rate slider (0.0001–0.1), momentum slider (0–0.99), surface selector dropdown (Rosenbrock / saddle / quadratic), reset button
6. Camera: OrbitControls for free rotation, plus a "top-down" button that animates to bird's eye (contour view)
7. Optional: show the gradient vector as an arrow (`ArrowHelper`) at the ball's position

---

### 2. HMC Puck on a Potential Energy Surface

**Post:** `hmcidea` (primary), `hmcexplore` (link to it)
**Priority:** Highest — HMC intuition is notoriously hard; this is the canonical visualization

**Description:** The negative log-posterior is shown as a 3D "potential energy" landscape. A frictionless puck is launched with random momentum (shown as velocity arrow). It traces leapfrog steps (discrete segments visible as dots connected by lines). At the end of the trajectory, an accept/reject coin flip. Side-by-side or toggle: a Metropolis random-walk particle doing small jittery steps on the same surface, for contrast. The HMC puck explores efficiently; the Metropolis walker gets stuck.

**Axes:** X = parameter 1, Y = parameter 2, Z (height) = negative log-posterior (potential energy)

**2D alternative:** Contour plot with two animated particles — one HMC (smooth arcs) and one Metropolis (jittery steps). Effective but loses the "rolling on a surface" physics intuition. D3.js with SVG paths.

**Preferred tech:** Three.js — the physics metaphor only works if you see the surface. The leapfrog trajectory curving along the surface contours is the key insight.

**Implementation sketch:**
1. Surface mesh from a banana-shaped posterior: `z = -logpdf(bivariate_normal_with_correlation)` or a funnel
2. Puck as a glowing sphere; momentum as an `ArrowHelper` that shrinks as kinetic energy converts to potential
3. Leapfrog integration: at each sub-step, compute gradient of surface, update momentum and position; show each step as a small sphere (breadcrumb trail)
4. Accept/reject: flash green (accept) or red (reject, snap back to start)
5. Controls: step size slider (large = divergence!), number of leapfrog steps slider, "launch" button for single trajectory, "auto-sample" toggle for continuous sampling
6. Comparison mode: split view or toggle between HMC and Metropolis on the same surface
7. Show accumulated samples as translucent dots building up the posterior shape

---

### 3. Metropolis Random Walk on a Posterior

**Post:** `metropolis` (primary), `metropolishastings`, `metropolissupport`
**Priority:** High — the foundational MCMC algorithm, builds intuition for everything after

**Description:** A translucent 3D posterior surface (bivariate normal, or banana-shaped). A glowing particle proposes a jump (shown as a dotted arc to a candidate point). If the candidate is higher on the surface, it's accepted (green flash); if lower, accepted probabilistically (amber flash) or rejected (red flash, snap back). The particle's history accumulates as a scatter of points that gradually reveals the posterior shape. With a wide proposal, many rejections; with narrow, slow diffusion.

**Axes:** X = parameter 1, Y = parameter 2, Z (height) = posterior density

**2D alternative:** Contour plot with animated particle and accept/reject flashes. Very effective in 2D honestly — the 3D version adds the "height = probability" intuition. Could do both: 2D as in-notebook widget, 3D as standalone page.

**Preferred tech:** Three.js for the full experience; but a **Canvas 2D** version embedded in the notebook (contour plot + particle) would also be valuable and lighter-weight. Consider building the 2D version first.

**Implementation sketch:**
1. Surface mesh from target distribution (bivariate normal with adjustable correlation)
2. Current position: bright sphere on the surface. Proposal: ghost sphere at candidate position connected by a dashed line
3. Animation sequence per step: (a) draw proposal line, (b) pause 300ms, (c) evaluate acceptance, (d) flash green/red, (e) move or snap back
4. Controls: proposal sigma slider (0.1–5.0), speed slider (steps per second), target selector (normal / banana / bimodal), reset button
5. Accumulated samples: small translucent spheres that persist, gradually building a point cloud
6. Stats overlay: acceptance rate counter, effective sample size
7. Camera: OrbitControls + "top-down" button for contour-like view

---

### 4. Gaussian Process Function Space

**Post:** `gp1` (primary), `gp2` (theory), `gp3` (inference)
**Priority:** High — GPs are famously hard to grok; seeing the function space is transformative

**Description:** The scene shows a "ribbon" of GP sample paths. X-axis is input, Y-axis is function value. Multiple translucent curves (sample functions from the prior) wave gently. Click on the X-axis to add an observation — all sample paths pinch through that point, and the uncertainty envelope (shaded band) narrows locally. Change the kernel and watch the samples change character: RBF gives smooth waves, Matern gives rougher paths, periodic gives oscillations.

**Axes:** X = input domain, Y = function value, Z = fanned-out sample paths (or just depth for visual separation of multiple draws)

**2D alternative:** This is actually very effective in 2D — a Canvas or D3.js plot with the mean line, confidence band, and ~10 sample paths. Clicking to add points triggers a smooth animation of the band narrowing. The 2D version might be *better* than 3D here since GPs are fundamentally 1D-in, 1D-out for intuition building.

**Preferred tech:** **D3.js or Canvas 2D** — the 2D version is more natural for this concept. Use smooth SVG path transitions for the "pinching" effect when observations are added. Three.js would only add value if you want to show the multivariate normal "tube" in 3D function space (X = input, Y = f(x), Z = orthogonal sample axis), which is cool but possibly confusing.

**Implementation sketch (2D with D3.js):**
1. Implement GP math in JS: kernel matrix K, Cholesky decomposition, conditional mean/variance formulas
2. Draw 10 sample paths from the prior as SVG paths with slight transparency
3. Draw mean line (solid) and +-2sigma band (shaded area)
4. On click: add observation to dataset, recompute posterior, animate all paths and band to new positions (D3 transitions, 500ms)
5. Controls: kernel selector (RBF, Matern-3/2, periodic), length-scale slider, noise variance slider, "clear observations" button
6. Show the kernel function itself in a small inset (K(x,x') as a function of distance)

---

### 5. KL Divergence Asymmetry

**Post:** `divergence` (primary), `entropy` (link)
**Priority:** High — KL asymmetry is one of the most confusing concepts; visualization makes it click

**Description:** Two distributions P (blue surface) and Q (orange surface) shown as 3D bell-shaped surfaces. A slider morphs Q's parameters (mean, variance) toward or away from P. Two KL values displayed: KL(P||Q) and KL(Q||P). The key insight: when Q is too narrow (misses P's tails), KL(P||Q) explodes; when Q is too wide (wastes mass), KL(Q||P) is moderate. Show the integrand `p(x) * log(p(x)/q(x))` as a third colored surface to see *where* the divergence comes from.

**Axes:** X = variable value, Y = density, Z = second variable (for 2D distributions) or used for visual separation of P vs Q

**2D alternative:** Actually very strong in 2D — two overlapping density curves with the KL integrand shaded underneath. Animate Q sliding/stretching. Dual KL values update in real-time. Probably better in 2D since distributions are 1D objects.

**Preferred tech:** **Canvas 2D or D3.js** — 1D distributions are best shown as curves, not surfaces. Use filled areas and smooth transitions. Three.js would only help for 2D (bivariate) distributions, which adds complexity without proportional insight.

**Implementation sketch (2D Canvas/D3):**
1. P = fixed Gaussian (or mixture of two Gaussians for more drama)
2. Q = Gaussian with adjustable mean and variance (two sliders)
3. Draw P as filled blue curve, Q as filled orange curve (semi-transparent overlap)
4. Below: draw the integrand `p(x) * log(p(x)/q(x))` as a red shaded area (this is where KL "comes from")
5. Display KL(P||Q) and KL(Q||P) as large numbers, color-coded by magnitude
6. "Swap direction" toggle: flips which divergence's integrand is shown
7. Preset buttons: "Q too narrow", "Q too wide", "Q matched" to jump to instructive configurations

---

## Priority 2: Strong Candidates

### 6. Rejection Sampling — Darts Under the Envelope

**Post:** `rejectionsampling`
**Priority:** Medium-high — very visual algorithm, intuitive "darts" metaphor

**Description:** Target density as a curve (or 3D surface for bivariate). Proposal envelope (scaled proposal density) as a taller curve above it. Animated darts fall from above at random x-positions. If the dart lands between the target and the x-axis, it's accepted (green, stays). If it lands between the target and the envelope, it's rejected (red, fades out). Accepted darts accumulate into a histogram that converges to the target.

**Axes:** X = variable value, Y = density (height where dart lands), Z = not needed (2D is natural)

**2D alternative:** This *is* a 2D concept. No 3D needed.

**Preferred tech:** **Canvas 2D** — fast rendering for many animated particles. D3 would work but Canvas is better for particle-heavy animation.

**Implementation sketch:**
1. Draw target density f(x) and envelope M*g(x) as filled curves
2. Each animation frame: sample x from g, sample u from Uniform(0, M*g(x)), draw a falling dot
3. If u < f(x): dot turns green, parks at (x, u), adds to histogram below
4. If u >= f(x): dot turns red, fades out over 500ms
5. Controls: envelope scale M slider (too tight = misses tails, too loose = high rejection), speed slider, proposal selector (uniform / Gaussian)
6. Stats: acceptance rate counter, histogram vs true density overlay
7. "Burst mode" button: drop 1000 darts at once to see the pattern

---

### 7. Importance Sampling — Weighted Particles

**Post:** `importancesampling`
**Priority:** Medium-high — weight degeneracy is hard to explain without seeing it

**Description:** Target distribution as a curve. Proposal distribution as a different curve. Particles sampled from the proposal, drawn as circles with radius proportional to their importance weight. When the proposal badly mismatches the target (e.g., too narrow), a few particles become enormous while most are tiny — effective sample size collapses visually. Shows why proposal choice matters.

**Axes:** X = variable value, Y = density, particle size = importance weight

**2D alternative:** This is inherently 2D. No 3D needed.

**Preferred tech:** **D3.js** — circles with smooth size transitions, good for interactive exploration.

**Implementation sketch:**
1. Draw target p(x) and proposal q(x) as overlapping filled curves
2. Sample N=100 particles from q(x), compute weights w_i = p(x_i)/q(x_i), normalize
3. Draw particles along x-axis with radius proportional to weight
4. Show effective sample size ESS = 1/sum(w_i^2) as a big number
5. Controls: proposal mean slider, proposal sigma slider, N particles slider, "resample" button
6. Animate: when sliders change, particles smoothly resize (D3 transitions)
7. Show weighted histogram vs true density — poor proposals give choppy histograms

---

### 8. Bayesian Updating — Prior x Likelihood Animation

**Post:** `bayes_withsampling` (primary), `globemodel` (Beta-Binomial version)
**Priority:** Medium-high — core Bayesian concept, sequential updating is powerful to watch

**Description:** Three curves stacked or overlaid: prior (blue), likelihood (gold), posterior (green). Click "observe heads" or "observe tails" (for the globe/coin model). The likelihood reshapes, and the posterior morphs in real-time as a smooth animation. After many observations, the posterior concentrates. A "reset" button returns to the flat prior.

**Axes:** X = parameter (e.g., probability of heads), Y = density

**2D alternative:** This *is* a 2D concept. The animation and interactivity are the key, not 3D.

**Preferred tech:** **D3.js** — smooth SVG path transitions are perfect for morphing density curves. The Beta distribution has a closed form, so no heavy computation.

**Implementation sketch:**
1. Prior: Beta(a, b) starting at Beta(1,1) = uniform
2. On "observe heads": a += 1, recompute posterior curve, animate transition (D3 path tween, 400ms)
3. On "observe tails": b += 1, same animation
4. Draw all three curves: prior (initial, stays fixed as reference), likelihood (binomial, reshapes), posterior (current Beta)
5. Controls: "observe heads" / "observe tails" buttons, "observe 10 random" button (rapid-fire), "reset" button, prior selector (uniform / informative / skeptical)
6. Show a, b values and posterior mean/mode as text
7. Optional: small coin-flip animation in corner when observing

---

### 9. EM Algorithm Convergence

**Post:** `em`
**Priority:** Medium — EM is visual but typically taught in 2D already

**Description:** 2D scatter plot of mixture data (two clusters). Two Gaussian bells shown as translucent ellipses (or 3D surfaces if going full Three.js). E-step: each point's color smoothly transitions to reflect its soft assignment (responsibility) to each cluster. M-step: the ellipses smoothly shift and reshape to fit their weighted points. Step through manually or auto-play.

**Axes (3D version):** X = feature 1, Y = feature 2, Z = density (Gaussian height)
**Axes (2D version):** X = feature 1, Y = feature 2, color = cluster responsibility

**2D alternative:** 2D scatter with colored ellipses is the standard and works great. The 3D version adds the density surface, which shows *why* the E-step assigns points the way it does (which bell is taller at that point).

**Preferred tech:** **Three.js for 3D** if paired with the loss surface demos (consistent tech); **D3.js for 2D** scatter + ellipses if keeping it lightweight. The 3D version is prettier; the 2D version is clearer.

**Implementation sketch (2D, D3.js):**
1. Generate 200 points from a 2-component Gaussian mixture (fixed dataset, randomizable)
2. Initialize two ellipses (random or k-means)
3. E-step: compute responsibilities, smoothly transition point colors (blue↔orange gradient by responsibility)
4. M-step: smoothly move and resize ellipses to weighted means and covariances
5. Controls: "E-step" / "M-step" buttons for manual stepping, "auto-play" toggle, "randomize init" button, speed slider
6. Show log-likelihood value increasing (line chart in corner)
7. Convergence: flash "converged!" when parameter change < epsilon

---

### 10. Variational Approximation Fitting

**Post:** `vi` (primary), `advi` (link)
**Priority:** Medium — makes the abstract ELBO optimization tangible

**Description:** True posterior shown as a complex contour (banana or bimodal). Variational approximation as a Gaussian ellipse that morphs during optimization. Mean-field: the ellipse can only be axis-aligned (can't capture correlations). Full-rank: the ellipse can rotate. Watch the approximation iterate toward the best fit. ELBO value climbs in a corner chart.

**Axes:** X = parameter 1, Y = parameter 2, contour height = density

**2D alternative:** Contour plot with animated ellipse. Very effective in 2D — this is how VI is typically visualized in papers.

**Preferred tech:** **Canvas 2D** or **D3.js** — contour plots + animated ellipses. Three.js only if you want to show the density surfaces in 3D (true posterior as a weird bumpy surface, Gaussian approximation as a smooth bell trying to match it).

**Implementation sketch (2D):**
1. True posterior: precomputed on a grid (banana: `log p ∝ -0.5*(x^2/σ^2 + (y-x^2)^2/τ^2)`)
2. Draw as filled contour plot (5-6 levels)
3. Variational Gaussian: parameterized by mean (μ1, μ2) and covariance matrix (3 params for full-rank, 2 for mean-field)
4. Gradient descent on negative ELBO (analytic for Gaussian target, use score function estimator for complex targets — or precompute trajectory and replay)
5. Each frame: draw the Gaussian as an ellipse overlay, show ELBO value
6. Controls: "mean-field" / "full-rank" toggle, "step" button, "auto-optimize" toggle, target selector (banana / bimodal / correlated normal)
7. Key moment: show mean-field failing to capture banana correlation (ellipse stays axis-aligned while posterior is tilted)

---

## Priority 3: Would Benefit, More Niche

### 11. Markov Chain State Space

**Post:** `markov`
**Priority:** Medium-low — visual but a graph problem, not really spatial

**Description:** States as labeled spheres arranged in a circle or force-directed layout. Edges as arrows with thickness proportional to transition probability. A glowing particle hops between states. After many steps, show the empirical visit frequency converging to the stationary distribution (bar chart overlay).

**Preferred tech:** **D3.js** — force-directed graph layout is D3's strength. Or **vis.js** / **cytoscape.js** for graph-specific features.

**Implementation sketch:**
1. Define transition matrix (3-5 states for clarity)
2. D3 force layout for node positions, arrow edges with width = P(i→j)
3. Particle: circle that smoothly moves along edges (D3 transition on arc path)
4. Bar chart below: visit counts per state, reference line for stationary distribution
5. Controls: speed slider, "step" button, transition matrix editor (advanced)

---

### 12. 3D Hit-or-Miss Monte Carlo

**Post:** `basicmontecarlo`
**Priority:** Medium-low — fun visual upgrade but the 2D version is already clear

**Description:** Estimate the volume of a sphere by throwing random points in a cube. Points inside the sphere glow green; outside glow red. The sphere is translucent. Running estimate of π (from volume ratio) converges.

**Preferred tech:** **Three.js** — the transparent sphere in a wireframe cube with accumulating particles is visually satisfying.

**Implementation sketch:**
1. Wireframe cube (`BoxGeometry` with `EdgesGeometry`)
2. Translucent sphere inside (`SphereGeometry`, opacity 0.2)
3. Each frame: add N random points as small spheres, color by inside/outside
4. Running estimate: `6 * (points_inside / total_points)` displayed
5. Controls: points-per-frame slider, reset button, OrbitControls

---

### 13. Neal's Funnel (Hierarchical Pathology)

**Post:** `gelmanschools` (link), `gelmanschoolstheory` (primary)
**Priority:** Medium-low — important concept but niche audience

**Description:** The funnel: X = group means, Y = log(variance), Z = density. At high variance (wide part), group means spread out. At low variance (narrow neck), they're tightly constrained. Centered parameterization: sampler particles get stuck in the neck. Non-centered: particles explore uniformly. Toggle between parameterizations.

**Preferred tech:** **Three.js** — the funnel is inherently 3D and the "stuck in the neck" phenomenon is visceral when you see it in 3D.

**Implementation sketch:**
1. Generate funnel surface: for each log(τ), the conditional distribution of θ is Normal(0, τ)
2. Show as a 3D surface or point cloud
3. Animate MCMC samples: centered (clusters in neck) vs non-centered (uniform spread)
4. Controls: parameterization toggle, "run sampler" button
5. Camera: allow orbit, plus preset views (side view to see the funnel shape)

---

### 14. Normal Model — Likelihood Surface

**Post:** `normalmodel`
**Priority:** Low — useful but standard

**Description:** 3D surface where X = mean, Y = variance, Z = posterior density. Click to add data points and watch the surface reshape. The key insight: the ridge along the variance axis is broad (variance is hard to pin down), while the mean direction sharpens quickly. With few data points the surface is a broad mesa; with many it collapses to a sharp peak.

**Axes:** X = mean (mu), Y = variance (sigma^2), Z = posterior density

**2D alternative:** Contour plot of the joint posterior over (mu, sigma^2) with clickable data entry. Contour lines show the ridge structure. Effective but the 3D view makes the "ridge vs peak" asymmetry more visceral.

**Preferred tech:** **Three.js** — the 3D surface is the natural representation. OrbitControls let you look along the ridge from different angles.

**Implementation sketch:**
1. Compute joint posterior on a (mu, sigma^2) grid using the Normal-InverseGamma conjugate: `p(mu, sigma^2 | data) ∝ (sigma^2)^{-(n/2+alpha+1)} * exp(-[beta + 0.5*sum((x_i - mu)^2)] / sigma^2)`
2. Surface mesh: `PlaneGeometry(100, 100)` with vertex heights set to posterior density values, color by height (sequential blue palette)
3. Click-to-add-data: click triggers a modal or number input, recomputes posterior grid, smoothly morphs vertex heights (lerp over 30 frames)
4. Show data points as small spheres along the mu axis at ground level (z=0)
5. Controls: "add data point" input, "reset" button, prior strength slider (adjusts alpha/beta hyperparameters)
6. Camera: OrbitControls + preset buttons for "top-down" (contour view) and "along ridge" (shows the elongation)

---

### 15. CLT Convergence Stack

**Post:** `samplingclt`
**Priority:** Low — elegant but the standard histogram animation works well

**Description:** Stacked distributions along the Z-axis where Z represents sample size n. At n=1, the raw population shape (could be skewed, bimodal, uniform). As n increases, the sampling distribution of the mean narrows and becomes Gaussian. Each layer is a translucent histogram ribbon. A slider for n smoothly interpolates between layers, or you can orbit around the stack to see the convergence from any angle.

**Axes:** X = sample mean value, Y = frequency/density, Z = sample size n (1, 2, 5, 10, 30, 100)

**2D alternative:** Single animated histogram/density that morphs as a slider changes n. Overlay a Gaussian reference curve. Show the standard deviation shrinking as 1/sqrt(n). This is simpler and arguably just as effective — the "stack" metaphor is elegant but the 2D morph conveys the same message.

**Preferred tech:** **Canvas 2D** for the simpler animated histogram (more accessible, lighter). Three.js only if you want the stacked-layers-in-space visual, which is more of a wow factor than a pedagogical necessity.

**Implementation sketch (2D Canvas):**
1. Define population distribution (selector: uniform, exponential, bimodal mixture, chi-squared)
2. For current n, simulate 10,000 sample means by drawing n values and averaging
3. Draw histogram of sample means (40 bins), overlay Gaussian PDF with mean=pop_mean, sd=pop_sd/sqrt(n)
4. Slider for n (1 to 200, log scale): on change, resimulate and smoothly morph bin heights (lerp over 20 frames)
5. Display current n, theoretical vs empirical SD, and a "normality score" (e.g., Shapiro-Wilk p-value)
6. Controls: population selector, n slider, "animate n" button (auto-sweeps n from 1 to 100)

**Implementation sketch (3D Three.js):**
1. For each n in [1, 2, 5, 10, 30, 100], precompute histogram of 10,000 sample means
2. Each histogram as a ribbon of `BoxGeometry` bars arranged at its Z-position (n value)
3. Translucent materials with color gradient (warm for small n, cool for large n)
4. Camera: OrbitControls, with a "side view" preset that shows the narrowing progression
5. Interactive: slider highlights one layer as opaque, dims others

---

### 16. Inverse Transform Sampling — CDF Projection

**Post:** `inversetransform`
**Priority:** Low — nice visual but the concept is straightforward

**Description:** The CDF curve drawn prominently on the right half of the canvas. On the left Y-axis, uniform random samples appear as dots. Each dot projects horizontally (animated ray, like the sun rays in the seasons demo) until it hits the CDF curve, then drops vertically to the X-axis. The X-axis accumulates samples into a histogram that gradually matches the target PDF (drawn as a reference curve below).

**Axes:** X = target distribution domain, Y = cumulative probability [0,1]

**2D alternative:** This *is* a 2D concept — no 3D version needed. The animation of the projection rays is what makes it work.

**Preferred tech:** **Canvas 2D** — fast rendering for many animated rays. D3.js would work but Canvas handles the per-frame ray animation more smoothly.

**Implementation sketch:**
1. Draw target CDF (e.g., exponential, normal, or beta) as a thick curve on the upper half of the canvas
2. Draw target PDF as a reference curve on the lower half
3. Each animation step: generate u ~ Uniform(0,1), draw a dot on the Y-axis at u
4. Animate a horizontal ray from (0, u) rightward until it intersects the CDF at x = F^{-1}(u)
5. Animate a vertical ray dropping from (x, u) down to (x, 0)
6. Place a permanent dot at x on the X-axis; update the accumulating histogram
7. Controls: speed slider (1–50 samples/sec), distribution selector (exponential, normal, beta), "burst 500" button, reset button
8. After enough samples, the histogram should visibly match the reference PDF — show a "match score"

---

### 17. Maximum Entropy on the Simplex

**Post:** `entropy`
**Priority:** Low — beautiful but abstract

**Description:** For a 3-outcome discrete distribution (p1, p2, p3), the probability simplex is an equilateral triangle (all valid distributions live on it). Entropy H = -sum(pi * log(pi)) is shown as height above the simplex. The result is a dome peaked at the center (uniform distribution = maximum entropy). Constraints like "E[X] = 1.5" appear as lines cutting across the simplex, and the constrained maximum entropy point is the highest point along that line.

**Axes:** X, Y = barycentric coordinates on the simplex (the triangle), Z = entropy value

**2D alternative:** Show the simplex as a 2D triangle with entropy as a heatmap (color intensity). Constraint lines drawn on top, with the max-entropy point marked. Effective but the 3D dome is more memorable.

**Preferred tech:** **Three.js** — the simplex-as-surface-in-3D is a natural and beautiful object. The dome shape is immediately intuitive.

**Implementation sketch:**
1. Create equilateral triangle mesh (subdivided, ~50x50 vertices) using barycentric coordinates
2. For each vertex (p1, p2, p3), compute H = -sum(pi * log(pi)), set Z = H. Handle pi=0 (H contribution = 0)
3. Color vertices by entropy value (warm = high, cool = low)
4. Add constraint planes: for a given expected value constraint E[X] = c, compute the line on the simplex where p1*x1 + p2*x2 + p3*x3 = c, render as a glowing line on the surface
5. Mark the maximum entropy point (highest Z on the constraint line) with a bright sphere
6. Controls: constraint value slider (adjusts E[X] from x_min to x_max), toggle constraint on/off, toggle between "dome view" and "top-down heatmap"
7. Camera: OrbitControls + preset for top-down (shows triangle with heatmap) and side view (shows dome)
8. Label the three vertices of the simplex with their corresponding degenerate distributions: (1,0,0), (0,1,0), (0,0,1)

---

## Priority 4: Possible, Lower Priority

### 18. Regularization Geometry

**Post:** `regularization`
**Priority:** Low

**Description:** The classic Tibshirani diagram in 3D: loss contour ellipsoids intersecting with the Ridge constraint surface (sphere) and Lasso constraint surface (diamond/octahedron). The intersection point is the regularized estimate. With Lasso, the diamond's corners touch the coordinate axes, producing exact zeros (sparsity). With Ridge, the sphere's smooth surface never touches the axes.

**Axes:** X = beta_1 (coefficient 1), Y = beta_2 (coefficient 2), Z = loss value (for contour surfaces) or just 2D with the constraint region shown

**2D alternative:** The standard 2D version (contour ellipses + diamond/circle constraint) is already the textbook figure. An interactive version where you drag the constraint boundary size and watch the solution point move would add real value even in 2D.

**Preferred tech:** **Three.js** for the 3D contour+constraint intersection; **D3.js** for the interactive 2D version. The 2D interactive version is probably more useful since the 3D version can be visually cluttered.

**Implementation sketch:**
1. Draw loss contour ellipses (centered at OLS solution) using parametric curves
2. Draw constraint region: circle (Ridge) or diamond (Lasso), size controlled by lambda slider
3. Highlight the point where the smallest contour touches the constraint — this is the regularized solution
4. Lambda slider: as lambda decreases, constraint region grows, solution moves toward OLS. As lambda increases, solution shrinks toward origin
5. Toggle: Ridge vs Lasso vs Elastic Net (rounded diamond)
6. Show coefficient values as a bar chart in a corner (watch beta_2 hit exactly zero with Lasso)
7. Controls: lambda slider, penalty selector, "animate lambda" button (sweeps from 0 to large)

---

### 19. Neural Network Output Surface

**Post:** `nnreg`
**Priority:** Low

**Description:** A 3D surface showing a neural network's output evolving during training. The surface starts as a random, wavy sheet (random weights). As training progresses, it warps to pass through the training data points (shown as fixed colored spheres). You can see the network interpolating between points and the surface gradually smoothing. Overfitting visible as wild oscillations between data points when trained too long.

**Axes:** X = input feature 1, Y = input feature 2, Z = network output (prediction)

**2D alternative:** For 1D input, show the network's output curve evolving over a scatter plot of training data. Simpler and arguably clearer for the regression case. D3.js with smooth path transitions.

**Preferred tech:** **Three.js** for the 2D-input surface version; **Canvas 2D** for the 1D-input curve version. The 1D version is more practical (easier to implement, clearer to read), but the 3D surface is more dramatic.

**Implementation sketch:**
1. Define a simple 2-layer MLP in JavaScript (forward pass only — precompute weight snapshots during training offline, or run training live with a small network)
2. At each training epoch, evaluate the network on a grid of (x, y) values, set Z = output
3. Surface mesh: `PlaneGeometry` with vertex heights from network output, colored by value
4. Training data: fixed spheres at (x_i, y_i, z_i) that the surface must approach
5. Animate through epochs: slider or auto-play, surface morphs from random to fitted
6. Controls: epoch slider (0 to 1000), learning rate display, loss curve in corner, "train more" button, "reset weights" button
7. Show overfitting: if you drag the epoch slider past the sweet spot, surface starts oscillating wildly between data points

---

### 20. Gibbs Sampling on a Correlated Posterior

**Post:** `tetchygibbs`, `gibbsconj`
**Priority:** Low

**Description:** A 2D contour plot of a highly correlated bivariate normal (long, narrow, tilted ellipse). A Gibbs sampler particle makes axis-aligned steps: horizontal move (sample x given y), then vertical move (sample y given x). With low correlation, it moves freely. With high correlation (r=0.99), the conditional distributions are narrow, so the particle takes tiny steps along the ridge and mixing is painfully slow. Compare with a joint sampler that moves diagonally.

**Axes:** X = parameter 1, Y = parameter 2, contours = posterior density

**2D alternative:** This *is* inherently a 2D concept. The axis-aligned staircase pattern on the contour plot is the whole insight.

**Preferred tech:** **Canvas 2D** — contour plot with animated particle and trail. Fast rendering for the step-by-step animation.

**Implementation sketch:**
1. Draw contour ellipses of bivariate normal with adjustable correlation r
2. Initialize particle at a starting point
3. Each Gibbs step: (a) sample x from p(x|y_current) — horizontal move, draw horizontal line to new x, (b) sample y from p(y|x_current) — vertical move, draw vertical line to new y
4. Trail: fading staircase path showing the characteristic zigzag pattern
5. Controls: correlation slider (0 to 0.99), speed slider, "step" button for manual stepping, "run 100 steps" button
6. Stats: show effective sample size, autocorrelation of the chain
7. Side panel: trace plots of x and y showing the slow random walk behavior at high correlation
8. Comparison mode: toggle to show a Metropolis sampler on the same target (can move diagonally, mixes faster)

---

### 21. Label Switching in Mixtures

**Post:** `mixtures_and_mcmc`
**Priority:** Low

**Description:** Two Gaussian mixture components shown as colored bell curves. During MCMC sampling, the component labels can swap — component 1 (blue) jumps to where component 2 (orange) was, and vice versa. A trace plot below shows the component means over MCMC iterations, with the classic "crossing caterpillar" pattern. An ordered constraint (mu_1 < mu_2) fixes the problem — toggle it on and the traces separate cleanly.

**Axes:** Main panel: X = data value, Y = density (bell curves). Trace panel: X = MCMC iteration, Y = component mean value.

**2D alternative:** This *is* a 2D concept — trace plots and density curves. No 3D needed.

**Preferred tech:** **D3.js** — smooth transitions for the bell curves swapping positions, and a scrolling trace plot below.

**Implementation sketch:**
1. Generate fixed dataset from a 2-component Gaussian mixture (well-separated)
2. Precompute (or simulate live) MCMC chain for mixture model parameters (mu_1, mu_2, sigma_1, sigma_2, pi)
3. Main panel: draw two bell curves colored by component, data histogram behind
4. Animate through MCMC iterations: bell curves shift positions according to sampled parameters
5. Trace panel below: scrolling line chart of mu_1 and mu_2 over iterations — show the crossing pattern
6. Controls: "run MCMC" button, speed slider, "apply ordering constraint" toggle (mu_1 < mu_2)
7. When ordering constraint is toggled on: re-sort components each iteration, trace lines separate cleanly
8. Stats: show marginal posterior of mu_1 (bimodal without constraint, unimodal with)

---

### 22. LKJ Correlation Prior — Deforming Ellipsoid

**Post:** `corr`
**Priority:** Low

**Description:** A 3D ellipsoid representing the correlation structure of a 3-variable system. The three axes of the ellipsoid correspond to the three pairwise correlations (r12, r13, r23). The LKJ eta parameter controls how concentrated the prior is around the identity matrix (no correlations). At eta=1, all valid correlation matrices are equally likely (the ellipsoid can be any shape). As eta increases, it shrinks toward a sphere (identity = no correlations). The visualization makes the abstract LKJ prior tangible.

**Axes:** The three principal axes of the ellipsoid represent the three variables; the ellipsoid's shape encodes the correlation matrix.

**2D alternative:** Show a grid of sampled correlation matrices as heatmaps (small multiples) that become more "identity-like" as eta increases. Or animate a single heatmap morphing. Less dramatic but clearer for understanding what correlation matrices actually look like.

**Preferred tech:** **Three.js** — the deforming 3D ellipsoid is visually compelling and maps directly to the geometric interpretation of correlation matrices.

**Implementation sketch:**
1. Generate a correlation matrix from the LKJ distribution for given eta (use Cholesky decomposition: sample from LKJ, compute C = L @ L.T)
2. Ellipsoid: `SphereGeometry` with a matrix transform applied — the correlation matrix defines the stretch/rotation. Use `mesh.matrix.set(...)` from the Cholesky factor
3. Animate: on eta slider change, sample several correlation matrices, interpolate ellipsoid shape (lerp on Cholesky factors)
4. Color the ellipsoid surface by local curvature or keep it translucent with wireframe overlay
5. Controls: eta slider (0.5 to 50, log scale), "sample new matrix" button (random draw from LKJ(eta)), "animate eta" button
6. Show the correlation matrix numerically as a heatmap in a corner panel
7. Camera: OrbitControls + preset views along each axis
8. At eta=1, show several random ellipsoids cycling to demonstrate the uniform-over-correlations property

---

### 23. Decision Boundary — Sigmoid Surface

**Post:** `logisticbp`
**Priority:** Low

**Description:** The logistic regression output P(class=1) as a 3D surface over the 2D feature space. The surface is a smooth sigmoid "ramp" — high on one side (class 1 region), low on the other (class 0 region), with the 0.5 contour line being the linear decision boundary. Data points sit as colored spheres on or near the surface (class 1 above, class 0 below). Rotating the view reveals both the surface shape and the boundary line.

**Axes:** X = feature 1, Y = feature 2, Z = P(class = 1) from sigmoid output

**2D alternative:** Top-down view with a colored decision region (blue vs orange background) and scatter points. Add the decision boundary as a line. This is the standard textbook figure — interactive version lets you drag points and watch the boundary shift.

**Preferred tech:** **Three.js** — the 3D sigmoid surface adds real insight beyond the standard 2D plot. Seeing the smooth ramp makes the "soft classification" nature of logistic regression tangible. The decision boundary is where the surface crosses z=0.5.

**Implementation sketch:**
1. Generate or load a 2D classification dataset (linearly separable or near-separable)
2. Fit logistic regression (precompute weights w1, w2, bias b, or fit live with gradient descent)
3. Surface: `PlaneGeometry(100, 100)` with vertex heights = sigmoid(w1*x + w2*y + b), colored by value (blue-to-orange diverging colormap)
4. Decision boundary: `Line` geometry at z=0.5 where sigmoid = 0.5 (a straight line in the XY plane, elevated to z=0.5)
5. Data points: spheres at (x_i, y_i, z_i) where z_i = actual class (0 or 1), colored by class
6. Controls: "train" button (animate gradient descent, watch surface tilt into place), learning rate slider, "add point" (click to place new data, retrain), weight display
7. Camera: OrbitControls + "top-down" preset (classic decision boundary view) + "side view" (shows sigmoid ramp)
8. Optional: show the loss landscape in a separate small panel (loss vs w1, w2) with the current position marked
