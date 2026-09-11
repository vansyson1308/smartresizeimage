# AutoBanner mission

**Objective.** Turn an approved master design into editable, coherent, accurate campaign
variants across sizes, content changes and markets, with substantially less setup and
correction than existing workflows.

**Commercial ambition.** A credible path to >= USD 1M ARR. This is a target to earn with
customer evidence, not a status. Engineering completion and commercial validation are tracked
as separate statuses in `CAPABILITIES.md`.

**Technical thesis.** A design is an executable, editable program: a typed representation with
stable element identities, native text, semantic roles, relationships/constraints, provenance
and confidence. Adaptation plans and renders from that representation; independent rendered
checks verify the result; repair is scoped; approved changes propagate with a reviewable diff.

## Initial customer hypothesis

Agencies and in-house creative teams producing repeated static advertising variants for several
brands, sizes and markets. Buyer: creative-ops lead, agency owner, marketing lead.
Status: hypothesis. No customer conversations have happened from this repository.

## Scope of the supported product (current definition)

- Inputs: layered PSD with native text layers, or a project created from separated assets.
  Flat PNG/JPG inputs are supported for adaptation but are explicitly marked "not decomposed"
  and always land in review.
- Outputs: raster PNG/JPEG variants per requested size plus a saved, reopenable project.
- Journey: import -> interpret/correct -> variant brief -> generate -> review/repair -> campaign
  change -> export/reopen.

## Authority and limits

Within the coding environment: inspect, test, implement, refactor, commit on the designated
branch, open draft PRs, install ordinary dependencies in an isolated venv.

Never: purchase services, raise spend, use customer assets for training, contact customers,
merge to a protected branch, change live billing, publicly launch, bypass access controls.

## Priorities (ordered)

1. Truthful evaluation: a score must describe what the customer sees.
2. One complete, reopenable end-to-end journey before breadth.
3. Native text and asset fidelity over generative novelty.
4. Measured differentiation (held-out evaluation, ablations) before claims.
5. Operability (isolation, jobs, metering) once the journey is real.
