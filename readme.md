## A Structured Multi-Stage Workflow for Controllable AI-Assisted Visual Creation

### Abstract

AI-generated concept images often present challenges for targeted refinement, particularly when multiple requirements regarding object identity, spatial relationships, and visual consistency must be satisfied simultaneously. Whole-image regeneration frequently corrects one requirement while introducing unintended disruptions to others.

This thesis proposes a dual-path workflow that enables selective reuse and structured refinement of concept imagery. The visual pathway extracts reusable visual elements from candidate sketches, while the structural pathway encodes object layout and relative depth information. These two representations are integrated via a fusion stage that explicitly distinguishes invariant properties from permitted appearance modifications. We present a complete evaluation framework spanning initial prompting strategies, element extraction, structural conditioning, fusion quality, and end-to-end joint constraint satisfaction under fixed model-call budgets.

We evaluate the proposed workflow across both controlled benchmark scenes and real-world concept sketch scenarios. Empirical results demonstrate key trade-offs between extraction region coverage and source fidelity, the complementary control roles of mask and depth conditioning, and the capacity of combined visual and spatial inputs to improve overall constraint compliance. Results also identify upstream extraction quality and automated evaluator reliability as critical performance-limiting factors. We further show that joint constraint satisfaction provides a stricter and more practically relevant measure of task success than average requirement scores alone.

The primary contributions of this work are the proposed workflow design and a systematic evaluation framework for studying controllable concept image refinement. Experimental findings validate the core design logic of the dual-path architecture and provide actionable insights for pipeline optimization. Full end-to-end validation on larger-scale datasets and user studies with intended creative practitioners remain important directions for future investigation.

