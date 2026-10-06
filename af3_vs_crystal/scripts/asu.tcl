# The whole asymmetric unit, one colour per copy.
#   vmd -dispdev text -e asu.tcl -args <asu.pdb> <scene.dat>
# 11_copy_example.py writes the copy number into the B-factor, in disjoint ranges,
# because a PDB chain id is one character and cannot carry it:
#     MHC of copy N -> N        peptide of copy N -> 10 + N
set inpdb  [lindex $argv 0]
set outdat [lindex $argv 1]
mol new $inpdb type pdb waitfor all
mol delrep 0 top

# MHC of each copy: cartoon, one ColorID per copy
foreach cp {1 2 3 4} col {0 3 7 10} {
    mol representation NewCartoon 0.32 12 4.1 0
    mol color ColorID $col
    mol selection "beta > [expr $cp - 0.5] and beta < [expr $cp + 0.5]"
    mol material AOChalky
    mol addrep top
}
# the peptide of each copy: thick licorice, red, so the three grooves are findable
mol representation Licorice 0.42 14 14
mol color ColorID 1
mol selection {beta > 10}
mol material AOShiny
mol addrep top

catch {display projection Orthographic}
catch {display depthcue off}
catch {axes location Off}
catch {color Display Background white}
catch {display ambientocclusion on}
catch {display aoambient 0.9}
catch {display aodirect 0.3}
catch {display shadows on}
display resetview
scale by 1.15
render Tachyon $outdat
quit
