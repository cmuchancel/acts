#!/usr/bin/env python3

from pathlib import Path
from typing import Optional
import argparse

import acts
from acts import UnitConstants as u
from acts.examples import GenericDetector, RootParticleReader
from acts.examples.odd import (
    getOpenDataDetector,
    getOpenDataDetectorDirectory,
)


def getArgumentParser():
    """Get arguments from command line"""
    parser = argparse.ArgumentParser(description="Command line arguments for CKF")
    parser.add_argument(
        "-i",
        "--indir",
        dest="indir",
        help="Directory with input root files",
        default="./",
    )

    parser.add_argument(
        "--geometry",
        choices=["generic", "odd"],
        default="generic",
        help="Detector geometry to use",
    )

    parser.add_argument(
        "-o",
        "--output",
        dest="outdir",
        help="Output directory for new ntuples",
        default="./",
    )
    parser.add_argument(
        "-n", "--nEvents", dest="nEvts", help="Number of events to run over", default=1
    )
    parser.add_argument(
        "--sf_maxSeedsPerSpM",
        dest="sf_maxSeedsPerSpM",
        help="Number of compatible seeds considered for middle seed",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--sf_cotThetaMax",
        dest="sf_cotThetaMax",
        help="cot of maximum theta angle",
        type=float,
        default=7.40627,
    )
    parser.add_argument(
        "--sf_sigmaScattering",
        dest="sf_sigmaScattering",
        help="How many sigmas of scattering to include in seeds",
        type=float,
        default=5,
    )
    parser.add_argument(
        "--sf_radLengthPerSeed",
        dest="sf_radLengthPerSeed",
        help="Average Radiation Length",
        type=float,
        default=0.1,
    )
    parser.add_argument(
        "--sf_impactMax",
        dest="sf_impactMax",
        help="max impact parameter in mm",
        type=float,
        default=3.0,
    )
    parser.add_argument(
        "--sf_maxPtScattering",
        dest="sf_maxPtScattering",
        help="maximum Pt for scattering cut in GeV",
        type=float,
        default=10.0,
    )
    parser.add_argument(
        "--sf_deltaRMin",
        dest="sf_deltaRMin",
        help="minimum value for deltaR separation in mm",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--sf_deltaRMax",
        dest="sf_deltaRMax",
        help="maximum value for deltaR separation in mm",
        type=float,
        default=60.0,
    )

    # new: geometry‑dependent minimum pT --------------------------------------
    parser.add_argument(
        "--sf_minPt",
        type=float,
        default=None,
        help="Minimum pT for seeding [GeV] – overrides geometry defaults",
    )

    parser.add_argument(
        "-P", "--pythia8",
        action="store_true",
        help="Generate events on the fly with Pythia8 (ttbar hard-process).",
    )

    parser.add_argument(
        "--ttbar-pu",
        type=int,
        default=0,
        help="Average pile-up ⟨μ⟩ to mix into the Pythia8 ttbar sample.",
    )

    parser.add_argument("--event-number", type=int, default=0,
                    help="index of the event to process from particles.root")

    return parser


def runCKFTracks(
    trackingGeometry,
    decorators,
    geometrySelection: Path,
    digiConfigFile: Path,
    field,
    outputDir: Path,
    NumEvents=1,
    truthSmearedSeeded=False,
    truthEstimatedSeeded=False,
    outputCsv=True,
    inputParticlePath: Optional[Path] = None,
    s=None,
    MaxSeedsPerSpM=1,
    CotThetaMax=7.40627,
    SigmaScattering=5,
    RadLengthPerSeed=0.1,
    ImpactMax=3.0,
    MaxPtScattering=10.0,
    DeltaRMin=1.0,
    DeltaRMax=60.0,
    MinPt=0.5,
    skipGun=False, 
):
    from acts.examples.simulation import (
        addParticleGun,
        EtaConfig,
        PhiConfig,
        ParticleConfig,
        addFatras,
        addDigitization,
        ParticleSelectorConfig,
        addDigiParticleSelection,
    )

    from acts.examples.reconstruction import (
        addSeeding,
        TrackSmearingSigmas,
        SeedFinderConfigArg,
        SeedFinderOptionsArg,
        SeedingAlgorithm,
        TruthEstimatedSeedingAlgorithmConfigArg,
        addCKFTracks,
    )

    s = s or acts.examples.Sequencer(
        events=int(NumEvents),
        numThreads=-1,
        logLevel=acts.logging.INFO,
        outputDir=outputDir,
    )
    for d in decorators:
        s.addContextDecorator(d)
    rnd = acts.examples.RandomNumbers(seed=42)
    outputDir = Path(outputDir)

    if inputParticlePath is not None:
        # ① read an existing particles.root
        acts.logging.getLogger("CKFExample").info(
            "Reading particles from %s", inputParticlePath.resolve()
        )
        s.addReader(
            RootParticleReader(
                level=acts.logging.INFO,
                filePath=str(inputParticlePath.resolve()),
                outputParticles="particles_generated",
                firstEvent=options.event_number,
                numEvents=1,
            )
        )

    elif not skipGun:
        # ② fallback particle gun (only when we didn’t ask to skip it)
        addParticleGun(
            s,
            EtaConfig(-2.0, 2.0),
            ParticleConfig(4, acts.PdgParticle.eMuon, True),
            PhiConfig(0.0, 360.0 * u.degree),
            multiplicity=2,
            rnd=rnd,
        )

    addFatras(
        s,
        trackingGeometry,
        field,
        rnd=rnd,
    )

    addDigitization(
        s,
        trackingGeometry,
        field,
        digiConfigFile=digiConfigFile,
        rnd=rnd,
    )

    addDigiParticleSelection(
        s,
        ParticleSelectorConfig(
            pt=(0.5 * u.GeV, None),
            measurements=(9, None),
            removeNeutral=True,
        ),
    )

    addSeeding(
        s,
        trackingGeometry,
        field,
        TrackSmearingSigmas(  # only used by SeedingAlgorithm.TruthSmeared
            # zero eveything so the CKF has a chance to find the measurements
            loc0=0,
            loc0PtA=0,
            loc0PtB=0,
            loc1=0,
            loc1PtA=0,
            loc1PtB=0,
            time=0,
            phi=0,
            theta=0,
            ptRel=0,
        ),
        SeedFinderConfigArg(
            r=(None, 200 * u.mm),  # rMin=default, 33mm
            deltaR=(DeltaRMin * u.mm, DeltaRMax * u.mm),
            collisionRegion=(-250 * u.mm, 250 * u.mm),
            z=(-2000 * u.mm, 2000 * u.mm),
            maxSeedsPerSpM=MaxSeedsPerSpM,
            cotThetaMax=CotThetaMax,
            sigmaScattering=SigmaScattering,
            radLengthPerSeed=RadLengthPerSeed,
            maxPtScattering=MaxPtScattering * u.GeV,
            minPt= MinPt * u.GeV,
            impactMax=ImpactMax * u.mm,
        ),
        SeedFinderOptionsArg(bFieldInZ=2 * u.T, beamPos=(0.0, 0, 0)),
        TruthEstimatedSeedingAlgorithmConfigArg(deltaR=(10.0 * u.mm, None)),
        seedingAlgorithm=(
            SeedingAlgorithm.TruthSmeared
            if truthSmearedSeeded
            else (
                SeedingAlgorithm.TruthEstimated
                if truthEstimatedSeeded
                else SeedingAlgorithm.GridTriplet
            )
        ),
        initialSigmas=[
            1 * u.mm,
            1 * u.mm,
            1 * u.degree,
            1 * u.degree,
            0 * u.e / u.GeV,
            1 * u.ns,
        ],
        initialSigmaQoverPt=0.1 * u.e / u.GeV,
        initialSigmaPtRel=0.1,
        initialVarInflation=[1.0] * 6,
        geoSelectionConfigFile=geometrySelection,
        outputDirRoot=outputDir,
        rnd=rnd,  # only used by SeedingAlgorithm.TruthSmeared
    )

    addCKFTracks(
        s,
        trackingGeometry,
        field,
        outputDirRoot=outputDir,
        outputDirCsv=outputDir / "csv" if outputCsv else None,
    )

    return s


if "__main__" == __name__:
    options = getArgumentParser().parse_args()

    geometry_choice = options.geometry

    Inputdir = options.indir
    Outputdir = options.outdir

    srcdir = Path(__file__).resolve().parent.parent.parent.parent

    # ------------------------------------------------------------------
    # Geometry‑specific setup
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Geometry selection
    # ------------------------------------------------------------------
    if options.geometry == "generic":
        detector = GenericDetector()
        default_minPt = 0.5  # GeV
        base_cfg_dir = Path(__file__).resolve().parents[3] / "Examples" / "Configs"
        geo_cfg = base_cfg_dir / "generic-seeding-config.json"
        digi_cfg     = base_cfg_dir / "generic-digi-smearing-config.json"
    else:  # ODD -------------------------------------------------------------
        odd_dir      = Path(getOpenDataDetectorDirectory())
        material_map = odd_dir / "data" / "odd-material-maps.root"
        mat_deco     = acts.IMaterialDecorator.fromFile(material_map)

        detector     = getOpenDataDetector(odd_dir=odd_dir,
                                        materialDecorator=mat_deco)  # identical to full_chain_odd
        default_pt   = 1.0  # GeV

        acts_dir     = Path(__file__).resolve().parents[3]
        geo_cfg      = acts_dir / "Examples/Configs/odd-seeding-config.json"
        digi_cfg     = acts_dir / "Examples/Configs/odd-digi-smearing-config.json"

    trackingGeometry = detector.trackingGeometry()
    decorators        = detector.contextDecorators()

    # ------------------------------------------------------------------
    # Build a base Sequencer *first* (so we can prep Pythia8 before Fatras)
    # ------------------------------------------------------------------
    seq_base = acts.examples.Sequencer(
        events=int(options.nEvts), numThreads=-1, logLevel=acts.logging.INFO, outputDir=str(Outputdir)
    )

    if options.pythia8:
        from acts.examples.simulation import addPythia8
        from acts.examples import RandomNumbers, GaussianVertexGenerator
        addPythia8(
            seq_base,
            hardProcess=["Top:qqbar2ttbar=on"],
            npileup=options.ttbar_pu,
            vtxGen=GaussianVertexGenerator(
                mean=acts.Vector4(0, 0, 0, 0),
                stddev=acts.Vector4(
                    0.0125 * u.mm,
                    0.0125 * u.mm,
                    55.5 * u.mm,
                    5.0 * u.ns,
                ),
            ),
            rnd=RandomNumbers(seed=42),
            outputDirRoot=Path(options.outdir),
            outputDirCsv=Path(options.outdir),
        )
    # ------------------------------------------------------------------
    # Resolve min‑pT cut
    # ------------------------------------------------------------------
    minPt_cut = options.sf_minPt if options.sf_minPt is not None else default_minPt

    field = acts.ConstantBField(acts.Vector3(0, 0, 2 * u.T))

    inputParticlePath = Path(Inputdir) / "particles.root"
    if not inputParticlePath.exists():
        inputParticlePath = None

    seq = runCKFTracks(
        trackingGeometry,
        decorators,
        field=field,
        geometrySelection = geo_cfg,
        digiConfigFile= digi_cfg,
        outputCsv=True,
        truthSmearedSeeded=False,
        truthEstimatedSeeded=False,
        inputParticlePath=inputParticlePath,
        outputDir=Outputdir,
        NumEvents=options.nEvts,
        MaxSeedsPerSpM=options.sf_maxSeedsPerSpM,
        CotThetaMax=options.sf_cotThetaMax,
        SigmaScattering=options.sf_sigmaScattering,
        RadLengthPerSeed=options.sf_radLengthPerSeed,
        ImpactMax=options.sf_impactMax,
        MaxPtScattering=options.sf_maxPtScattering,
        DeltaRMin=options.sf_deltaRMin,
        DeltaRMax=options.sf_deltaRMax,
        MinPt=minPt_cut,
        skipGun=options.pythia8 ,
        s = seq_base,
    )
        
    seq_base.run()