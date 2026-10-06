# Overlay panel: predicted peptide (opaque) on crystal peptide (transparent),
# inside the crystal's MHC groove (ghost).
#   vmd -dispdev text -e overlay.tcl -args <panel.pdb> <scene.dat>
# Chains are written by 07_prepare_overlays.py: M = groove, P = crystal, Q = prediction.
set inpdb  [lindex $argv 0]
set outdat [lindex $argv 1]

mol new $inpdb type pdb waitfor all
mol delrep 0 top

# --- MHC groove: faint context, never the subject of the figure ---
mol representation NewCartoon 0.30 12 4.1 0
mol color ColorID 6
mol selection {chain M}
mol material Ghost
mol addrep top

# --- crystal peptide: TRANSPARENT, drawn thicker so it reads as a shell ---
mol representation Licorice 0.38 14 14
mol color ColorID 0
mol selection {chain P}
mol material Transparent
mol addrep top

# --- predicted peptide: OPAQUE and thinner, so it sits inside the crystal's shell ---
mol representation Licorice 0.24 14 14
mol color ColorID 1
mol selection {chain Q}
mol material AOShiny
mol addrep top

catch {display projection Orthographic}
catch {display depthcue off}
catch {axes location Off}
catch {color Display Background white}
catch {display ambientocclusion on}
catch {display aoambient 0.85}
catch {display aodirect 0.35}
catch {display shadows on}

display resetview
scale by 1.25

render Tachyon $outdat
quit
