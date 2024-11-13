import ROOT
fileName = "/hpc/group/vossenlab/rck32/eic/eicKLMcluster/examples/hepmc_5events_test_file_0.edm4hep.root"
file = ROOT.TFile(fileName)
tree = file.Get("events")

for event in tree:
    for MCParticle in event.MCParticles:
        print(type(MCParticle.parents_begin))
        print(f"MCParticle PDG, parent:{MCParticle.PDG},{MCParticle.parents_begin}")
