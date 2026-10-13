# Producing a Bessel beam with an SLM

Research restart — 12 October 2026. The previous single-plane imaging and phase-diversity experiments have been removed. Start with the physical SLM-to-beam transformation before rebuilding imaging or correction experiments.

## What produces the shape?

A zeroth-order ideal Bessel field is

\[
E(r,z)=E_0 J_0(k_\perp r)\exp(i k_z z),\qquad
k_\perp=k\sin\theta,\quad k_z=k\cos\theta.
\]

Its intensity has a bright central core surrounded by concentric rings. The field is a coherent superposition of plane waves on a cone: their transverse wavevectors occupy a ring with a common phase. A finite beam/aperture produces a quasi-Bessel beam over a finite axial interval. A ring-shaped intensity image alone does not establish the required field phase.

There are two useful SLM layouts. The displayed pattern must be designed for the plane in which the SLM actually sits.

## Route A: display a digital axicon

For a collimated beam incident on a phase-only SLM, start with a radial phase ramp:

\[
\phi_\mathrm{ax}(x,y)=-k_\perp\sqrt{(x-x_0)^2+(y-y_0)^2}.
\]

With propagation convention exp(i kz), the negative radial slope points rays inward for positive propagation distance. Reflection, relay inversion, and software conventions must be checked experimentally. The SLM displays this phase modulo 2π: concentric **sawtooth phase rings**, not a binary annular aperture and not a grayscale Bessel intensity image.

An optional carrier separates the useful order from unmodulated light:

\[
\phi_\mathrm{SLM}=\operatorname{mod}\left[
-k_\perp r+2\pi(f_x x+f_y y)+\phi_\mathrm{cal}(x,y),\;2\pi\right].
\]

Here x,y,r are physical distances at the SLM, f_x,f_y are cycles per unit distance, and φ_cal compensates measured static phase errors. Use a wavelength-specific phase-to-gray lookup table. A linear gray-to-phase relation is only an illustration unless calibrated. For a bright on-axis J0 core, start without a vortex term; adding ℓ atan2(y−y0,x−x0) produces higher-order modes with a dark center for nonzero integer ℓ.

Layout: collimated Gaussian → SLM → propagation/appropriate relay → Bessel region. If a carrier is used, include a Fourier-plane order-selection stage and account for its relay magnification. The Fourier plane of an axicon field is approximately a ring; the extended Bessel core belongs to a different plane/propagation region. A single FFT image of a phase axicon is therefore insufficient to demonstrate Bessel-beam formation.

Müller et al. experimentally generate Bessel arrays with SLM axicon phase masks [1]. Chattrapiban et al. demonstrate programmable multilevel SLM holograms and controllable beam size and deflection [2].

### Useful design estimates (scalar, paraxial where indicated)

The radial phase-wrap spacing is Λ_r=2π/k_perp. Therefore sin θ=λ/(n Λ_r) in a homogeneous medium with k=2πn/λ. In air take n≈1. Do not substitute an air cone angle unchanged after refraction or an objective relay.

The first zero of J0 gives central-core radius r_zero≈2.4048/k_perp; the intensity FWHM diameter is approximately 2.2527/k_perp. These are different width definitions. For a hard illuminated radius R, the geometric axial range is approximately z_max≈R/tan θ. Gaussian illumination weights that range, so it does not predict a flat axial profile or a measured axial FWHM. These estimates are derived from the ideal field and ray geometry, rather than fitted to our old notebooks.

A steeper ramp shrinks the core and shortens the geometrical range at fixed R. A wider incident beam increases the available range. Pixel sampling matters: for pixel pitch p, the radial ramp alone needs Λ_r>2p to avoid Nyquist aliasing, and appreciably more pixels per wrap are desirable. With a carrier, check both Cartesian components of the **combined** local spatial frequency against 1/(2p). The radial and carrier budgets cannot be assessed independently.

## Route B: use the SLM to synthesize an annular objective pupil

For a microscope, a practical alternative is:

collimated Gaussian → phase-only SLM → Fourier lens → coherent annular field + spatial filter → relay to objective back pupil → objective → extended Bessel-like focus.

The SLM is Fourier-conjugate to the ring plane. It need not be conjugate to the objective pupil. A digital axicon is a simple initial way to direct light into a ring, but its ring width and complex phase should be checked. If that field does not meet the target, use phase-only hologram design with amplitude **and phase** constraints on the annulus. Specifying only ring intensity leaves phase freedom that can spoil the central core.

Walde et al. demonstrate an iterative Fourier algorithm targeting a uniform-phase ring, with surrounding dark regions and a physical baffle [3]. This is evidence that a phase-only SLM can generate the required intermediate field; it is not a claim that a phase-only device directly blocks the center of its own illumination. Their broadband-specific algorithm is an option for later, rather than a prerequisite for the first monochromatic axicon trial.

### Relating SLM pattern to the microscope

For Fourier-lens focal length f1 in air, an axicon ramp of period Λ_r produces approximately ring radius ρ1=f1 λ/Λ_r (paraxial). With ring-plane-to-pupil magnification M, ρ_p=|M|ρ1. Using the objective sine condition, NA_ring≈ρ_p/f_obj and k_perp=(2π/λ_vac) NA_ring. Thus the ideal core radius is approximately 0.3827 λ_vac/NA_ring. Confirm the objective's actual pupil scale and clipping limits before using this relation.

An annulus with finite radial thickness spans axial wavevectors k_z=√(k²−k_perp²), limiting axial persistence. Narrower spectral rings generally extend the focus, but the actual axial profile depends on ring amplitude/phase and illumination. At high NA or for multiphoton excitation, use vector diffraction and the appropriate intensity power for final predictions; scalar J0 width estimates are starting points.

## Recommended next experiment

First identify the existing SLM plane and relay. If free propagation or an image relay is available, begin with Route A and a calibrated wrapped radial ramp. If the goal is annular illumination of the microscope objective, explicitly model Route B from the SLM through the Fourier lens and pupil relay.

1. Record vacuum wavelength, SLM pixel pitch/resolution, usable phase stroke/LUT, illuminated beam radius and polarization, lens positions/focal lengths, relay magnification, objective NA/index and pupil size.
2. Compute the physical phase pattern, including carrier and calibration, and verify sampling and pupil acceptance.
3. Propagate the complex illuminated SLM field through the actual layout. Track power lost to filtering and clipping.
4. Inspect intensity **and phase** at the annulus, then measure/simulate transverse profiles at multiple axial positions. Compare core width, side lobes, axial intensity, and transmitted power.
5. Only after this check, connect the delivered pupil field to the existing vector/multiphoton PSF model. Return to phase diversity after generation is established.

No hardware values have been assumed here and no display-ready SLM image has been generated. The immediate missing information is the SLM's optical conjugacy and the relay/objective geometry.

## Sources

[1] Müller et al., *Efficient generation of Bessel beam arrays by means of an SLM* (2011). Author-institution record: https://eprints.gla.ac.uk/59759/ . Supports the digital-axicon route.

[2] Chattrapiban et al., *Generation of nondiffracting Bessel beams by use of a spatial light modulator*, Optics Letters 28, 2183–2185 (2003): https://doi.org/10.1364/OL.28.002183 . Supports programmable multilevel phase holograms and beam control.

[3] Walde et al., *Engineering an achromatic Bessel beam using a phase-only spatial light modulator and an iterative Fourier transformation algorithm*, Optics Communications 383, 64–68 (2017). Open full text: https://oa.tib.eu/renate/server/api/core/bitstreams/ee7e5803-fca5-44a2-b960-de4f4cfbe69e/content . Supports coherent annular-field synthesis and explicit intermediate-plane filtering.
