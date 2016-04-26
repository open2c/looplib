## Simulate loop extrusion by loop extruding factors and analyse the resulting structures of loops.

- simlef.pyx - simulate loop extrusion in 1d with simulate()

- looptools.py - analyse the structure of extruded loops:
    - build trees of nested loops with get_parent_loops()
    - find root loops with get_roots()
    - identify the positions between the root loops with get_backbone()
    - identify stacked LEFs with stack_lefs()
    
- loopviz.py - visualize loops with plot_lefs()

- simlef_twosided.pyx - simulate loops extrusion with synchronized extrusion blocking

