# MaxEntVFP

A MaximumEntropy approach according to [Sivia1991](https://www.sciencedirect.com/science/article/abs/pii/092145269190042D)

Creates a flexible ('freeform') profile which is constrained by the interfacial volume of material
(itself a fittable parameter). The profile itself can be constrained to be monotonic. This model
was designed to replicate the structure of a polymer brush, but would be well suited to any diffuse
interface where the interfacial volume can be approximated.

Here the volume fraction profile is described by a series of pixels. The volume fraction
of polymer in these pixels is allowed to vary. The smoothness of the profile is encouraged by adding
an entropic term to the log-posterior probability.


Added by:
Andrew Nelson

Date: 07/08/2023