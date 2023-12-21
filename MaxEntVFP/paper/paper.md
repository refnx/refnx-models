- slowly varying SLD profile, e.g. a ramp, possibly either side of a bulk medium
- a sharpish peak?
- part of the component that is negative (entropy only works on positive values)
- volume fraction profile of a brush
- non-uniform profile from recent articles

Papers
------


Notes
-----
- smearing doesn't work if the MaxEnt component needs to have steep changes at its extremes, e.g. to join to other components with very different SLDs. The smearing convolution isn't fully supported for cells at the end, so one has to decide how the smearing is handled for the end cells. One can use e.g. mirroring/padding/etc, but these can artificially affect the SLD at the edges. Perhaps it's a better idea to only smear in the middle section where the convolution is fully supported.
